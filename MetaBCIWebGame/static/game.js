// game.js - 脑控游戏前端（双Canvas + 灰度刺激 + 丢帧监测 + 评测模式）
(function() {
    // ==================== 常量 ====================
    const WS_URL = `ws://${window.location.hostname}:8765`;
    const MAZE_DEFAULT_W = 31, MAZE_DEFAULT_H = 31;
    const DEMO_TIMEOUT_MS = 3000;
    const DEMO_MAX_RETRIES = 10;
    const WS_RECONNECT_DELAY = 3000;

    const stimPhases = {
        up: 0,      // 0 * π
        down: 0.5,  // 0.5 * π
        left: 1,    // 1 * π
        right: 1.5  // 1.5 * π
    };
    const stimFreqs = { up: 8.25, down: 11.0, left: 13.75, right: 16.5 };
    const dirKeys = ['up', 'down', 'left', 'right'];
    const dirToIdx = { up: 0, down: 1, left: 2, right: 3 };
    const idxToDir = ['up', 'down', 'left', 'right'];

    // ==================== DOM 元素 ====================
    const gameCanvas = document.getElementById('gameCanvas');
    const gameCtx = gameCanvas.getContext('2d');
    const stimCanvas = document.getElementById('stimCanvas');
    const stimCtx = stimCanvas.getContext('2d');

    const lastCmdSpan = document.getElementById('last-cmd');
    const demoLogDiv = document.getElementById('demo-log');
    const demoSummaryDiv = document.getElementById('demo-summary');
    const demoProgressDiv = document.getElementById('demo-progress');
    const wsStatusSpan = document.getElementById('ws-status');

    // ==================== 全局状态 ====================
    let currentGame = 'maze';
    let currentMode = 'offline';
    let realtimeActive = false;
    let ws = null;
    let wsReconnectTimer = null;
    let activeGame = null;

    // 离线演示相关
    let demoActive = false;
    let demoPath = [];
    let demoCurrentStep = 0;
    let demoActualSteps = [];
    let demoTimeoutId = null;
    let demoRetryCount = 0;
    let demoStopFlag = false;

    // 评测模式
    let evalMode = false;
    let evalTarget = null;
    let evalSequence = [];
    let evalTrialIndex = 0;
    let evalWaitingForResult = false;
    let evalPromptTimer = null;      // 提示阶段定时器
    let evalMainTimer = null;        // 解码超时定时器
    let evalExtraTimer = null;       // 额外保护超时
    let evalRestTimer = null;
    let showIndicator = true;
    let evalResults = [];
    let evalRetryCount = 0;
    let evalTrialStartTime = 0;

    // ==================== 刺激控制 ====================
    let stimFlashing = false;
    let stimAnimationId = null;
    let stimStartTime = null;
    let lastStimFrameTime = 0;
    let frameIntervals = [];

    // ==================== 刺激块布局 ====================
    const STIM_CONFIG = {
        blockWidth: 150,
        blockHeight: 150,
        gap: 65,
    };

    let positions = {};

    function initStimPositions() {
        const cw = stimCanvas.width;
        const ch = stimCanvas.height;
        const totalW = STIM_CONFIG.blockWidth * 4 + STIM_CONFIG.gap * 3;
        const startX = (cw - totalW) / 2;
        const baseY = (ch - STIM_CONFIG.blockHeight) / 2;
        const dirOrder = ['up', 'down', 'left', 'right'];
        const pos = {};
        dirOrder.forEach((dir, i) => {
            const x = startX + i * (STIM_CONFIG.blockWidth + STIM_CONFIG.gap) + STIM_CONFIG.blockWidth / 2;
            const y = baseY + STIM_CONFIG.blockHeight / 2;
            pos[dir] = { x, y, w: STIM_CONFIG.blockWidth, h: STIM_CONFIG.blockHeight };
        });
        return pos;
    }
    positions = initStimPositions();

    // ==================== 刺激绘制核心 ====================
    function drawStimuli(now) {
        stimCtx.clearRect(0, 0, stimCanvas.width, stimCanvas.height);

        const isFlashing = stimFlashing && stimStartTime !== null;
        const t = isFlashing ? (now - stimStartTime) / 1000 : 0;

        for (const dir of dirKeys) {
            let gray;
            if (isFlashing) {
                const freq = stimFreqs[dir];
                const phase = stimPhases[dir] * Math.PI;
                const val = 0.5 + 0.5 * Math.sin(2 * Math.PI * freq * t + phase);
                gray = Math.floor(128 + 127 * val);
            } else {
                gray = 128;
            }

            const pos = positions[dir];
            if (!pos) continue;
            const x = pos.x - pos.w/2;
            const y = pos.y - pos.h/2;
            stimCtx.fillStyle = `rgb(${gray}, ${gray}, ${gray})`;
            stimCtx.fillRect(x, y, pos.w, pos.h);
            stimCtx.fillStyle = (gray > 128) ? '#000' : '#fff';
            stimCtx.font = '28px Arial';
            stimCtx.textAlign = 'center';
            stimCtx.textBaseline = 'middle';
            const label = {up:'↑', down:'↓', left:'←', right:'→'}[dir];
            stimCtx.fillText(label, pos.x, pos.y);
        }

        if (evalMode && evalTarget && showIndicator) {
            drawEvalIndicator(stimCtx, evalTarget);
        }
    }

    // ==================== 评测指示标（水平箭头） ====================
    function drawEvalIndicator(ctx, targetDir) {
        if (!targetDir || !positions[targetDir]) return;

        const dirOrder = ['up', 'down', 'left', 'right'];
        const idx = dirOrder.indexOf(targetDir);
        if (idx === -1) return;

        let leftDir, rightDir;
        if (idx <= 1) {
            leftDir = 'up';
            rightDir = 'down';
        } else {
            leftDir = 'left';
            rightDir = 'right';
        }

        const leftPos = positions[leftDir];
        const rightPos = positions[rightDir];
        const centerX = (leftPos.x + rightPos.x) / 2;
        const centerY = leftPos.y;

        const isLeft = (targetDir === leftDir);
        const direction = isLeft ? -1 : 1;

        ctx.save();
        ctx.fillStyle = '#ff0000';
        ctx.shadowBlur = 12;
        ctx.shadowColor = '#ff0000';

        const arrowSize = 28;
        const tipX = centerX + direction * arrowSize * 0.6;
        const tipY = centerY;
        ctx.beginPath();
        ctx.moveTo(tipX, tipY);
        ctx.lineTo(tipX - direction * arrowSize * 0.8, tipY - arrowSize * 0.6);
        ctx.lineTo(tipX - direction * arrowSize * 0.8, tipY + arrowSize * 0.6);
        ctx.closePath();
        ctx.fill();
        ctx.restore();
    }

    // ==================== 刺激动画循环 ====================
    function animateStim(now) {
        if (stimFlashing) {
            const interval = now - lastStimFrameTime;
            lastStimFrameTime = now;
            frameIntervals.push(interval);
            if (frameIntervals.length > 300) frameIntervals.shift();
            if (frameIntervals.length === 60) {
                const dropped = frameIntervals.filter(v => v > 30).length;
                const dropRate = dropped / frameIntervals.length;
                if (dropRate > 0.05) {
                    console.warn(`[刺激] 丢帧率 ${(dropRate*100).toFixed(1)}%`);
                }
                frameIntervals = [];
            }
        }
        drawStimuli(now);
        stimAnimationId = requestAnimationFrame(animateStim);
    }

    // ==================== 对外控制接口 ====================
    function startStimuli() {
        if (!stimFlashing) {
            stimFlashing = true;
            stimStartTime = performance.now();
            lastStimFrameTime = stimStartTime;
            frameIntervals = [];
            if (!stimAnimationId) {
                stimAnimationId = requestAnimationFrame(animateStim);
            }
        }
    }

    function stopStimuli() {
        stimFlashing = false;
        drawStimuli(performance.now());
    }

    // ==================== 游戏基类 ====================
    class BaseGame {
        init() {}
        handleMove(cmd) {}
        render(ctx) {}
        getScore() { return 0; }
        getDiamonds() { return { collected: 0, total: 0 }; }
    }

    // ==================== 迷宫游戏（完整实现） ====================
    class MazeGame extends BaseGame {
        constructor() { super(); this.state = null; }
        init(width = MAZE_DEFAULT_W, height = MAZE_DEFAULT_H) {
            const maze = this._generateMazePrim(width, height);
            const [exitX, exitY] = this._findFarthestPoint(maze, 1, 1);
            this._connectExitToBoundary(maze, exitX, exitY);
            maze[exitY][exitX] = 0;
            const totalCells = ((width-1)/2)*((height-1)/2);
            let dCount = Math.max(3, Math.floor(totalCells/30));
            dCount = Math.min(dCount, totalCells-2);
            const diamonds = this._generateDiamonds(maze, dCount);
            const cellSize = Math.min(gameCanvas.width / maze[0].length, gameCanvas.height / maze.length);
            this.state = {
                maze, player: [1,1], goal: [exitY,exitX], score: 0, moves: 0,
                cell_size: cellSize, diamonds, collectedDiamonds: 0, totalDiamonds: dCount
            };
            this._updateUI();
        }
        _generateMazePrim(width, height) {
            if (width%2===0) width++; if (height%2===0) height++;
            const maze = Array(height).fill().map(()=>Array(width).fill(1));
            let startX = 1+2*Math.floor(Math.random()*(width-1)/2);
            let startY = 1+2*Math.floor(Math.random()*(height-1)/2);
            maze[startY][startX] = 0;
            const walls = [];
            const addWalls = (x,y) => {
                const dirs = [[-2,0],[2,0],[0,-2],[0,2]];
                for (let [dx,dy] of dirs) {
                    let nx = x+dx, ny = y+dy;
                    if (nx>0 && nx<width-1 && ny>0 && ny<height-1 && maze[ny][nx]===1) {
                        walls.push({ wx: x+dx/2, wy: y+dy/2, nx, ny });
                    }
                }
            };
            addWalls(startX, startY);
            while (walls.length) {
                const idx = Math.floor(Math.random()*walls.length);
                const wall = walls[idx];
                if (maze[wall.ny][wall.nx] === 1) {
                    maze[wall.wy][wall.wx] = 0;
                    maze[wall.ny][wall.nx] = 0;
                    addWalls(wall.nx, wall.ny);
                }
                walls.splice(idx,1);
            }
            maze[1][0] = 0;
            return maze;
        }
        _findFarthestPoint(maze, sx, sy) {
            const h=maze.length, w=maze[0].length;
            const dirs=[[0,1],[1,0],[0,-1],[-1,0]];
            const dist=Array(h).fill().map(()=>Array(w).fill(Infinity));
            const q=[[sx,sy]]; dist[sy][sx]=0;
            while(q.length) {
                const [x,y]=q.shift();
                for(let [dx,dy] of dirs) {
                    const nx=x+dx, ny=y+dy;
                    if(nx>=0 && nx<w && ny>=0 && ny<h && maze[ny][nx]===0 && dist[ny][nx]===Infinity) {
                        dist[ny][nx]=dist[y][x]+1;
                        q.push([nx,ny]);
                    }
                }
            }
            let max=-1, best=[sx,sy];
            for(let i=1;i<h-1;i++) for(let j=1;j<w-1;j++) if(maze[i][j]===0 && dist[i][j]>max) { max=dist[i][j]; best=[j,i]; }
            return best;
        }
        _connectExitToBoundary(maze, ex, ey) {
            const h=maze.length, w=maze[0].length;
            if(ex===w-2) maze[ey][w-1]=0;
            else if(ex===1) maze[ey][0]=0;
            else if(ey===h-2) maze[h-1][ex]=0;
            else if(ey===1) maze[0][ex]=0;
            else {
                const toTop=ey, toBottom=h-1-ey, toLeft=ex, toRight=w-1-ex;
                const minDist=Math.min(toTop,toBottom,toLeft,toRight);
                if(minDist===toTop) for(let y=ey; y>=0; y--) maze[y][ex]=0;
                else if(minDist===toBottom) for(let y=ey; y<h; y++) maze[y][ex]=0;
                else if(minDist===toLeft) for(let x=ex; x>=0; x--) maze[ey][x]=0;
                else for(let x=ex; x<w; x++) maze[ey][x]=0;
            }
        }
        _generateDiamonds(maze, count) {
            const h=maze.length, w=maze[0].length;
            const cand=[];
            for(let i=1;i<h-1;i++) for(let j=1;j<w-1;j++) if(maze[i][j]===0 && !(i===1&&j===1)) cand.push([i,j]);
            for(let i=cand.length-1;i>0;i--) { const j=Math.floor(Math.random()*(i+1)); [cand[i],cand[j]]=[cand[j],cand[i]]; }
            return cand.slice(0, Math.min(count, cand.length));
        }
        handleMove(cmd) {
            if (!this.state) return;
            let newPos = [...this.state.player];
            switch(cmd) {
                case 'up': newPos[0]--; break;
                case 'down': newPos[0]++; break;
                case 'left': newPos[1]--; break;
                case 'right': newPos[1]++; break;
                default: return;
            }
            if(newPos[0]<0 || newPos[0]>=this.state.maze.length || newPos[1]<0 || newPos[1]>=this.state.maze[0].length) return;
            if(this.state.maze[newPos[0]][newPos[1]] === 0) {
                const idx = this.state.diamonds.findIndex(d => d[0]===newPos[0] && d[1]===newPos[1]);
                if(idx !== -1) {
                    this.state.diamonds.splice(idx,1);
                    this.state.collectedDiamonds++;
                    this.state.score += 10;
                    this._updateUI();
                }
                this.state.player = newPos;
                this.state.moves++;
                if(newPos[0]===this.state.goal[0] && newPos[1]===this.state.goal[1]) {
                    const rate = (this.state.collectedDiamonds/this.state.totalDiamonds)*100;
                    let bonus = 10;
                    let msg = "收集率不足50%";
                    if(rate===100) { bonus=50; msg="完美收集！"; }
                    else if(rate>=80) { bonus=30; msg="收集率超过80%"; }
                    else if(rate>=50) { bonus=20; msg="收集率超过50%"; }
                    this.state.score += bonus;
                    this._updateUI();
                    alert(`到达出口！\n钻石: ${this.state.collectedDiamonds}/${this.state.totalDiamonds} (${Math.round(rate)}%)\n${msg}\n+${bonus}分\n总分:${this.state.score}`);
                    if (evalMode) {
                        stopEvalMode();
                    } else if (demoActive) {
                        stopDemo();
                    } else {
                        const w = this.state.maze[0].length;
                        const h = this.state.maze.length;
                        this.init(w, h);
                    }
                    return;
                }
                this.render(gameCtx);
            }
        }
        render(ctx) {
            const s = this.state;
            if (!s) return;
            const cs = s.cell_size;
            const offX = (gameCanvas.width - s.maze[0].length*cs)/2;
            const offY = (gameCanvas.height - s.maze.length*cs)/2;
            ctx.clearRect(0,0,gameCanvas.width,gameCanvas.height);
            for(let i=0;i<s.maze.length;i++) {
                for(let j=0;j<s.maze[i].length;j++) {
                    const x = offX + j*cs, y = offY + i*cs;
                    ctx.fillStyle = s.maze[i][j]===1 ? '#2c2c2c' : '#d9d9d9';
                    ctx.fillRect(x, y, cs, cs);
                }
            }
            for(let d of s.diamonds) {
                const x = offX + d[1]*cs, y = offY + d[0]*cs;
                const cx = x+cs/2, cy = y+cs/2, r = cs*0.3;
                ctx.beginPath(); ctx.moveTo(cx,cy-r); ctx.lineTo(cx+r,cy); ctx.lineTo(cx,cy+r); ctx.lineTo(cx-r,cy); ctx.fillStyle='#ffcc00'; ctx.fill();
                ctx.fillStyle='#ffaa00'; ctx.font=`${cs*0.4}px "Segoe UI"`; ctx.fillText("💎", cx-cs*0.18, cy+cs*0.15);
            }
            const gx = offX + s.goal[1]*cs, gy = offY + s.goal[0]*cs;
            ctx.fillStyle='#ffaa44'; ctx.fillRect(gx,gy,cs,cs);
            ctx.fillStyle='white'; ctx.font=`${cs*0.5}px "Segoe UI"`; ctx.fillText("⭐", gx+cs*0.25, gy+cs*0.7);
            const px = offX + s.player[1]*cs + cs/2, py = offY + s.player[0]*cs + cs/2;
            ctx.beginPath(); ctx.arc(px,py,cs*0.4,0,Math.PI*2); ctx.fillStyle='#44ff44'; ctx.shadowBlur=8; ctx.fill(); ctx.shadowBlur=0;
        }
        getScore() { return this.state ? this.state.score : 0; }
        getDiamonds() {
            if (!this.state) return { collected: 0, total: 0 };
            return { collected: this.state.collectedDiamonds, total: this.state.totalDiamonds };
        }
        _updateUI() {
            const scoreSpan = document.getElementById('score');
            if(scoreSpan) scoreSpan.innerText = this.state.score;
            const collectedSpan = document.getElementById('collectedDiamonds');
            const totalSpan = document.getElementById('totalDiamonds');
            if(collectedSpan) collectedSpan.innerText = this.state.collectedDiamonds;
            if(totalSpan) totalSpan.innerText = this.state.totalDiamonds;
        }
        recomputeShortestPath() {
            if (!this.state) return [];
            const maze = this.state.maze;
            const start = this.state.player;
            const goal = this.state.goal;
            const h = maze.length, w = maze[0].length;
            const dirs = [[0, -1, 'up'], [0, 1, 'down'], [-1, 0, 'left'], [1, 0, 'right']];
            const queue = [{x: start[1], y: start[0], path: []}];
            const visited = Array(h).fill().map(()=>Array(w).fill(false));
            visited[start[0]][start[1]] = true;
            while (queue.length) {
                let {x, y, path} = queue.shift();
                if (x === goal[1] && y === goal[0]) return path;
                for (let [dx, dy, dir] of dirs) {
                    let nx = x + dx, ny = y + dy;
                    if (nx>=0 && nx<w && ny>=0 && ny<h && maze[ny][nx]===0 && !visited[ny][nx]) {
                        visited[ny][nx] = true;
                        queue.push({x: nx, y: ny, path: [...path, dir]});
                    }
                }
            }
            return [];
        }
    }

    // ==================== 贪吃蛇游戏（完整） ====================
    class SnakeGame extends BaseGame {
        constructor() { super(); this.state = null; }
        init() {
            this.state = {
                snake: [[5,5], [5,4], [5,3]],
                direction: 'right',
                food: [5,7],
                score: 0
            };
            this._updateUI();
        }
        handleMove(cmd) {
            if (!this.state) return;
            let newDir = null;
            switch(cmd) {
                case 'up': newDir = 'up'; break;
                case 'down': newDir = 'down'; break;
                case 'left': newDir = 'left'; break;
                case 'right': newDir = 'right'; break;
                default: return;
            }
            const opposite = {up:'down', down:'up', left:'right', right:'left'};
            if (newDir !== opposite[this.state.direction]) {
                this.state.direction = newDir;
                let head = this.state.snake[0];
                let newHead = [...head];
                switch(this.state.direction) {
                    case 'up': newHead[0]--; break;
                    case 'down': newHead[0]++; break;
                    case 'left': newHead[1]--; break;
                    case 'right': newHead[1]++; break;
                }
                this.state.snake.unshift(newHead);
                if(newHead[0]===this.state.food[0] && newHead[1]===this.state.food[1]) {
                    this.state.score += 10;
                    this.state.food = [Math.floor(Math.random()*20), Math.floor(Math.random()*20)];
                    this._updateUI();
                } else {
                    this.state.snake.pop();
                }
                this.render(gameCtx);
            }
        }
        render(ctx) {
            const cs = 20;
            const offX = (gameCanvas.width - 20*cs)/2, offY = (gameCanvas.height - 20*cs)/2;
            ctx.clearRect(0,0,gameCanvas.width,gameCanvas.height);
            for(let seg of this.state.snake) {
                ctx.fillStyle = '#33ff33';
                ctx.fillRect(offX + seg[1]*cs, offY + seg[0]*cs, cs-1, cs-1);
            }
            ctx.fillStyle = '#ff3333';
            ctx.fillRect(offX + this.state.food[1]*cs, offY + this.state.food[0]*cs, cs-1, cs-1);
        }
        getScore() { return this.state ? this.state.score : 0; }
        _updateUI() {
            const scoreSpan = document.getElementById('score');
            if(scoreSpan) scoreSpan.innerText = this.state.score;
            const collectedSpan = document.getElementById('collectedDiamonds');
            const totalSpan = document.getElementById('totalDiamonds');
            if(collectedSpan) collectedSpan.innerText = 0;
            if(totalSpan) totalSpan.innerText = 0;
        }
    }

    // ==================== 赛车游戏（完整） ====================
    class RacingGame extends BaseGame {
        constructor() { super(); this.state = null; }
        init() {
            this.state = { position: 0.5, speed: 5, score: 0 };
            this._updateUI();
        }
        handleMove(cmd) {
            if (!this.state) return;
            switch(cmd) {
                case 'left': this.state.position = Math.max(0, this.state.position - 0.05); break;
                case 'right': this.state.position = Math.min(1, this.state.position + 0.05); break;
                case 'up': this.state.speed = Math.min(20, this.state.speed + 1); break;
                case 'down': this.state.speed = Math.max(1, this.state.speed - 1); break;
            }
            this.state.score += Math.floor(this.state.speed);
            this._updateUI();
            this.render(gameCtx);
        }
        render(ctx) {
            ctx.clearRect(0,0,gameCanvas.width,gameCanvas.height);
            const roadW = gameCanvas.width * 0.6;
            const carW = 40;
            const carX = gameCanvas.width/2 - carW/2 + (this.state.position-0.5)*(roadW-carW);
            ctx.fillStyle = '#555';
            ctx.fillRect(0, gameCanvas.height-100, gameCanvas.width, 100);
            ctx.fillStyle = '#ff0000';
            ctx.fillRect(carX, gameCanvas.height-80, carW, 60);
        }
        getScore() { return this.state ? this.state.score : 0; }
        _updateUI() {
            const scoreSpan = document.getElementById('score');
            if(scoreSpan) scoreSpan.innerText = this.state.score;
            const collectedSpan = document.getElementById('collectedDiamonds');
            const totalSpan = document.getElementById('totalDiamonds');
            if(collectedSpan) collectedSpan.innerText = 0;
            if(totalSpan) totalSpan.innerText = 0;
        }
    }

    // ==================== 游戏管理 ====================
    const gameInstances = {
        maze: new MazeGame(),
        snake: new SnakeGame(),
        racing: new RacingGame()
    };

    function switchGame(gameType) {
        if (demoActive) stopDemo();
        if (evalMode) stopEvalMode();
        currentGame = gameType;
        document.querySelectorAll('#offline-panel .game-selector button').forEach(b => b.classList.remove('active'));
        document.getElementById('btn-maze').classList.toggle('active', gameType==='maze');
        document.getElementById('btn-snake').classList.toggle('active', gameType==='snake');
        document.getElementById('btn-racing').classList.toggle('active', gameType==='racing');
        document.querySelectorAll('#online-panel .game-selector button').forEach(b => b.classList.remove('active'));
        document.getElementById('btn-maze-online').classList.toggle('active', gameType==='maze');
        document.getElementById('btn-snake-online').classList.toggle('active', gameType==='snake');
        document.getElementById('btn-racing-online').classList.toggle('active', gameType==='racing');
        const mazeControls = document.getElementById('maze-controls');
        if (mazeControls) mazeControls.style.display = (gameType === 'maze') ? 'flex' : 'none';
        activeGame = gameInstances[gameType];
        activeGame.init();
        activeGame.render(gameCtx);
    }

    function switchGameOnline(gameType) {
        if (currentMode === 'online') switchGame(gameType);
    }

    // ==================== 离线演示 ====================
    function stopDemo() {
        if (!demoActive) return;
        if (demoTimeoutId) clearTimeout(demoTimeoutId);
        demoActive = false;
        demoStopFlag = true;
        if (ws && ws.readyState === WebSocket.OPEN) {
            ws.send(JSON.stringify({type: "stop_demo"}));
        }
        const startBtn = document.getElementById('btn-demo-start');
        const stopBtn = document.getElementById('btn-demo-stop');
        if (startBtn) startBtn.disabled = false;
        if (stopBtn) stopBtn.disabled = true;

        const totalAttempts = demoActualSteps.length;
        let correct = 0;
        for (let s of demoActualSteps) {
            if (s.match) correct++;
        }
        const accuracy = totalAttempts > 0 ? (correct / totalAttempts * 100).toFixed(2) : 0;
        if (demoSummaryDiv) {
            demoSummaryDiv.innerHTML = `总尝试: ${totalAttempts} | 正确步数: ${correct} | 准确率: ${accuracy}%`;
        }
        if (demoProgressDiv) {
            demoProgressDiv.innerHTML = `原始路径: ${demoPath.length} 步，实际尝试: ${totalAttempts} 次`;
        }
        if (demoLogDiv) {
            demoLogDiv.innerHTML += `\n🏁 演示结束。总尝试 ${totalAttempts}，正确 ${correct}，准确率 ${accuracy}%`;
        }
    }

    async function startDemo() {
        if (demoActive) return;
        if (evalMode) stopEvalMode();
        if (currentGame !== 'maze' || !(activeGame instanceof MazeGame)) {
            alert("请先切换到迷宫游戏");
            return;
        }
        const mazeGame = activeGame;
        const path = mazeGame.recomputeShortestPath();
        if (path.length === 0) {
            alert("无法找到路径！");
            return;
        }
        await startEvalMode(path);
    }

    function sendNextDemoStep() {
        if (!demoActive || demoStopFlag) return;
        if (demoCurrentStep >= demoPath.length) {
            setTimeout(() => {
                if (demoActive) stopDemo();
            }, 2000);
            return;
        }
        const expected = demoPath[demoCurrentStep];
        if (ws && ws.readyState === WebSocket.OPEN) {
            ws.send(JSON.stringify({type: "demo_step", direction: expected}));
            if (demoTimeoutId) clearTimeout(demoTimeoutId);
            demoTimeoutId = setTimeout(() => {
                if (demoActive && !demoStopFlag) {
                    recordDemoStep({
                        expected: expected,
                        decoded: 'timeout',
                        match: false,
                        filename: '超时',
                        confidence: 0,
                        all_confidences: [0,0,0,0]
                    });
                }
            }, DEMO_TIMEOUT_MS);
        } else {
            setTimeout(sendNextDemoStep, 500);
        }
    }

    function recordDemoStep(result) {
        if (result.all_confidences) {
            updateConfidenceBars(result.all_confidences);
        }
        if (demoTimeoutId) clearTimeout(demoTimeoutId);
        if (!demoActive || demoStopFlag) return;

        const expected = result.expected;
        const decoded = result.decoded;
        const match = result.match;
        const filename = result.filename || '未知文件';
        const conf = (result.confidence * 100).toFixed(1);

        const stepNum = demoCurrentStep + 1;
        demoActualSteps.push({
            expected: expected,
            decoded: decoded,
            match: match,
            filename: filename,
            confidence: conf,
            step: stepNum,
            retry: demoRetryCount
        });

        const totalAttempts = demoActualSteps.length;
        if (demoProgressDiv) {
            demoProgressDiv.innerHTML = `步骤: ${stepNum} / ${demoPath.length} (尝试 ${totalAttempts} 次)`;
        }

        if (demoLogDiv) {
            const statusIcon = match ? '✅' : '❌';
            const retryInfo = demoRetryCount > 0 ? ` (重试${demoRetryCount})` : '';
            const logMsg = `步骤 ${stepNum}${retryInfo}: 期望 ${expected} → 实际 ${decoded} ${statusIcon} (文件: ${filename})`;
            demoLogDiv.innerHTML += logMsg + '\n';
            demoLogDiv.scrollTop = demoLogDiv.scrollHeight;
        }

        if (match) {
            handleLocalMove(expected);
            if (demoCurrentStep >= demoPath.length - 1) {
                setTimeout(() => {
                    if (demoActive) stopDemo();
                }, 500);
            } else {
                demoCurrentStep++;
                demoRetryCount = 0;
                setTimeout(sendNextDemoStep, 500);
            }
        } else {
            demoRetryCount++;
            if (demoRetryCount >= DEMO_MAX_RETRIES) {
                if (demoLogDiv) {
                    demoLogDiv.innerHTML += `⚠️ 步骤 ${stepNum} 跳过（超过重试上限）\n`;
                }
                demoCurrentStep++;
                demoRetryCount = 0;
                setTimeout(sendNextDemoStep, 500);
            } else {
                setTimeout(sendNextDemoStep, 500);
            }
        }
    }

    // ==================== 评测模式（重构，与离线演示逻辑一致） ====================
    async function startEvalMode(sequence, source = 'offline') {
    if (demoActive) stopDemo();
    if (!sequence) {
        if (currentGame !== 'maze' || !(activeGame instanceof MazeGame)) {
            alert("请先切换到迷宫游戏");
            return;
        }
        const mazeGame = activeGame;
        const path = mazeGame.recomputeShortestPath();
        if (path.length === 0) {
            alert("无法找到路径！");
            return;
        }
        sequence = path;
    }
    evalSequence = sequence;
    evalTrialIndex = 0;
    evalMode = true;
    evalWaitingForResult = false;
    showIndicator = true;
    evalResults = [];
    evalRetryCount = 0;

    // 根据当前模式决定发送的消息类型
    const msgType = (currentMode === 'online') ? "start_eval" : "start_offline_sim";
    if (ws && ws.readyState === WebSocket.OPEN) {
        ws.send(JSON.stringify({ type: msgType }));
        console.log(`[前端] 发送 ${msgType}`);
        try {
            await new Promise((resolve, reject) => {
                const timeout = setTimeout(() => {
                    ws.removeEventListener('message', handler);
                    reject(new Error('等待 eval_started 超时'));
                }, 10000);
                const handler = (event) => {
                    try {
                        const data = JSON.parse(event.data);
                        if (data.type === "eval_started") {
                            clearTimeout(timeout);
                            ws.removeEventListener('message', handler);
                            resolve();
                        } else if (data.type === "eval_error") {
                            clearTimeout(timeout);
                            ws.removeEventListener('message', handler);
                            reject(new Error(data.message));
                        }
                    } catch (e) {}
                };
                ws.addEventListener('message', handler);
            });
            console.log('[前端] 后端已进入评测模式');
        } catch (e) {
            console.warn('[前端] 启动评测失败:', e.message);
            alert('启动评测失败: ' + e.message);
            evalMode = false;   // 重置状态
            document.getElementById('btn-eval-start').disabled = false;
            document.getElementById('btn-eval-stop').disabled = true;
            document.getElementById('btn-demo-start').disabled = false;
            document.getElementById('btn-demo-stop').disabled = true;
            return;
        }
    } else {
        alert('WebSocket 未连接');
        return;
    }

    if (!stimFlashing) startStimuli();

    document.getElementById('eval-info').style.display = 'block';
    document.getElementById('btn-eval-start').disabled = true;
    document.getElementById('btn-eval-stop').disabled = false;
    document.getElementById('eval-info').innerHTML = `评测准备就绪，共 ${sequence.length} 步`;
    document.getElementById('btn-demo-start').disabled = true;
    document.getElementById('btn-demo-stop').disabled = false;

    setTimeout(() => { if (evalMode) startNextEvalTrial(); }, 500);
}

    function stopEvalMode() {
    evalMode = false;
    evalTarget = null;
    evalWaitingForResult = false;
    showIndicator = true;
    clearTimeout(evalPromptTimer);
    clearTimeout(evalMainTimer);
    clearTimeout(evalExtraTimer);
    clearTimeout(evalRestTimer);
    document.getElementById('eval-info').style.display = 'none';
    document.getElementById('btn-eval-start').disabled = false;
    document.getElementById('btn-eval-stop').disabled = true;
    document.getElementById('btn-demo-start').disabled = false;
    document.getElementById('btn-demo-stop').disabled = true;

    if (stimFlashing) stopStimuli();
    if (realtimeActive) stopRealtime();
    if (activeGame) activeGame.render(gameCtx);

    const totalAttempts = evalResults.length;
    let correct = 0;
    for (let s of evalResults) if (s.match) correct++;
    const accuracy = totalAttempts > 0 ? (correct / totalAttempts * 100).toFixed(2) : 0;
    // 显示在 demo-summary 区域
    const summaryDiv = document.getElementById('demo-summary');
    if (summaryDiv) {
        summaryDiv.innerHTML = `在线评测结果 | 总尝试: ${totalAttempts} | 正确步数: ${correct} | 准确率: ${accuracy}%`;
    }
    console.log(`[评测汇总] 总尝试: ${totalAttempts}, 正确: ${correct}, 准确率: ${accuracy}%`);
}

    function startNextEvalTrial() {
        if (evalTrialIndex >= evalSequence.length) {
            stopEvalMode();
            alert('✅ 评测完成！请查看控制台获取详细记录。');
            return;
        }
        evalTarget = evalSequence[evalTrialIndex];
        evalWaitingForResult = false;
        showIndicator = true;
        evalRetryCount = 0;
        document.getElementById('eval-info').innerHTML = `📊 步骤 ${evalTrialIndex+1}/${evalSequence.length} | 👀 请注视 ${evalTarget}`;
        sendEvalTrialStart();
    }

    // ==================== 核心：发送评测步骤 ====================
function sendEvalTrialStart() {
    if (!evalMode) return;

    // 防止重复发送：如果已有等待结果，则忽略本次调用
    if (evalWaitingForResult) {
        console.warn('[前端] 已有等待结果，不重复发送 eval_step');
        return;
    }

    showIndicator = true;
    document.getElementById('eval-info').innerHTML = `📊 步骤 ${evalTrialIndex+1}/${evalSequence.length} | 👀 请注视 ${evalTarget} (尝试 ${evalRetryCount+1})`;

    // 清除之前可能残留的定时器
    clearTimeout(evalPromptTimer);
    clearTimeout(evalMainTimer);
    clearTimeout(evalExtraTimer);

    // 1.5秒提示阶段
    evalPromptTimer = setTimeout(() => {
        if (!evalMode) return;
        showIndicator = false;
        document.getElementById('eval-info').innerHTML = `📊 步骤 ${evalTrialIndex+1}/${evalSequence.length} | 🧠 解码中... (2秒)`;

        // 发送 eval_step
        if (ws && ws.readyState === WebSocket.OPEN) {
            ws.send(JSON.stringify({ type: "eval_step", direction: evalTarget }));
            console.log(`[前端] 发送 eval_step, expected=${evalTarget}, 重试=${evalRetryCount}`);
        } else {
            console.error('[前端] WebSocket 未连接，无法发送 eval_step');
            handleEvalResult({ match: false, decoded: 'ws_error', confidence: 0 });
            return;
        }
        evalWaitingForResult = true;
        evalTrialStartTime = Date.now();

        // 主超时：5000ms（5秒）
        evalMainTimer = setTimeout(() => {
            if (evalWaitingForResult) {
                console.warn('[前端] 等待 eval_result 超时，继续等待后端响应...');
                // 额外1.5秒保护
                evalExtraTimer = setTimeout(() => {
                    if (evalWaitingForResult) {
                        console.warn('[前端] 最终超时，强制失败');
                        handleEvalResult({ match: false, decoded: 'timeout', confidence: 0 });
                    }
                }, 1500);
            }
        }, 5000);
    }, 1500);
}

    // ==================== 处理评测结果 ====================
function handleEvalResult(data) {
    // 移除基于 step 的去重逻辑
    if (!evalMode) {
        console.log('[前端] 评测模式未激活，忽略 eval_result');
        return;
    }

    // 如果不在等待状态，忽略延迟到达的结果，防止状态错位
    if (!evalWaitingForResult) {
        console.warn('[前端] 收到 eval_result 但未在等待状态（可能超时后延迟到达），忽略');
        return;
    }

    evalWaitingForResult = false;

    // 清除所有定时器
    clearTimeout(evalPromptTimer);
    clearTimeout(evalMainTimer);
    if (evalExtraTimer) {
        clearTimeout(evalExtraTimer);
        evalExtraTimer = null;
    }

    // 更新置信度条
    if (data.all_confidences) {
        updateConfidenceBars(data.all_confidences);
    }

    const match = data.match;
    const decoded = data.decoded || '无';
    const confidence = (data.confidence * 100).toFixed(1);
    const expected = evalTarget;

    // 记录结果（不再用 step 去重，每收到一次就记录）
    evalResults.push({
        expected: expected,
        decoded: decoded,
        match: match,
        confidence: data.confidence || 0,
        step: evalTrialIndex + 1,
        retry: evalRetryCount
    });

    const statusIcon = match ? '✅' : '❌';
    document.getElementById('eval-info').innerHTML = `📊 步骤 ${evalTrialIndex+1}/${evalSequence.length} | ${statusIcon} (解码: ${decoded}, ${confidence}%)`;
    console.log(`[评测] 步骤 ${evalTrialIndex+1}, 尝试 ${evalRetryCount+1}: 期望=${expected}, 解码=${decoded}, 匹配=${match}`);

    if (match) {
        if (activeGame) {
            activeGame.handleMove(expected);
            activeGame.render(gameCtx);
        }
        evalTrialIndex++;
        evalRetryCount = 0;
        clearTimeout(evalRestTimer);
        evalRestTimer = setTimeout(() => {
            if (evalMode) startNextEvalTrial();
        }, 1000);
    } else {
        evalRetryCount++;
        if (evalRetryCount >= DEMO_MAX_RETRIES) {
            console.warn(`[评测] 步骤 ${evalTrialIndex+1} 跳过（超过重试上限 ${DEMO_MAX_RETRIES}）`);
            document.getElementById('eval-info').innerHTML = `📊 步骤 ${evalTrialIndex+1}/${evalSequence.length} |⚠️ 跳过（超过重试上限）`;
            evalTrialIndex++;
            evalRetryCount = 0;
            clearTimeout(evalRestTimer);
            evalRestTimer = setTimeout(() => {
                if (evalMode) startNextEvalTrial();
            }, 1000);
        } else {
            document.getElementById('eval-info').innerHTML = `📊 步骤 ${evalTrialIndex+1}/${evalSequence.length} | 🔄 重试 (${evalRetryCount}/${DEMO_MAX_RETRIES})`;
            clearTimeout(evalRestTimer);
            evalRestTimer = setTimeout(() => {
                if (evalMode) sendEvalTrialStart();
            }, 500);
        }
    }
}

    // ==================== 统一移动入口 ====================
    function handleLocalMove(cmd, fromWebSocket = false) {
        if (evalMode && !fromWebSocket) return;
        if (activeGame) {
            activeGame.handleMove(cmd);
            activeGame.render(gameCtx);
        }
    }

    // ==================== 模式切换 ====================
    function switchMode(mode) {
        if (currentMode === mode) return;
        currentMode = mode;
        document.getElementById('mode-offline').classList.toggle('active', mode === 'offline');
        document.getElementById('mode-online').classList.toggle('active', mode === 'online');
        document.getElementById('offline-panel').style.display = mode === 'offline' ? 'block' : 'none';
        document.getElementById('online-panel').style.display = mode === 'online' ? 'block' : 'none';
        if (ws && ws.readyState === WebSocket.OPEN) ws.send(JSON.stringify({type: "mode_switch", mode: mode}));
        if (mode === 'offline' && realtimeActive) stopRealtime();
        else if (mode === 'online' && realtimeActive) startRealtime();
    }

    // ==================== WebSocket 管理 ====================
    function updateWSStatus(connected) {
        if (wsStatusSpan) {
            wsStatusSpan.innerHTML = connected ? '🔌 WebSocket: 已连接' : '🔌 WebSocket: 未连接';
            wsStatusSpan.style.color = connected ? '#4caf50' : '#f44336';
        }
    }

    function connectWebSocket() {
        if (ws && (ws.readyState === WebSocket.OPEN || ws.readyState === WebSocket.CONNECTING)) return;
        ws = new WebSocket(WS_URL);
        ws.onopen = () => {
            console.log('WebSocket 已连接');
            updateWSStatus(true);
            if (wsReconnectTimer) clearTimeout(wsReconnectTimer);
            ws.send(JSON.stringify({type: "mode_switch", mode: currentMode}));
            if (currentMode === 'online' && realtimeActive) ws.send(JSON.stringify({type: "start_realtime"}));
        };
        ws.onmessage = (event) => {
    try {
        const data = JSON.parse(event.data);
        console.log('[前端] 收到 WebSocket 消息:', data);
        if (data.type === "demo_result") {
            recordDemoStep(data);
        } else if (data.type === "trial_result") {
            handleEvalResult(data);
        } else if (data.type === "eval_result") {      // ⭐ 新增
            handleEvalResult(data);
        } else if (data.type === "realtime_command") {
            if (lastCmdSpan) lastCmdSpan.innerText = data.command;
            if (data.all_confidences) updateConfidenceBars(data.all_confidences);
            if (!evalMode && !demoActive) {
                handleLocalMove(data.command, true);
            }
        } else if (data.type === "eval_started") {
            console.log('[前端] 后端已进入评测模式');
        } else if (data.type === "trigger_ack") {
            console.log('[Trigger] 已发送:', data.code);
        } else if (data.type === "offline_status" || data.type === "realtime_status") {
            // ignore
        } else if (data.command) {
            // 兼容旧格式
            if (lastCmdSpan) lastCmdSpan.innerText = data.command;
            if (data.all_confidences) updateConfidenceBars(data.all_confidences);
            if (!evalMode && !demoActive) {
                handleLocalMove(data.command, true);
            }
        }
    } catch (e) {
        console.error('WebSocket 消息解析错误:', e);
    }
};
        ws.onclose = () => {
            console.warn('WebSocket 断开，3秒后重连');
            updateWSStatus(false);
            if (wsReconnectTimer) clearTimeout(wsReconnectTimer);
            wsReconnectTimer = setTimeout(connectWebSocket, WS_RECONNECT_DELAY);
        };
        ws.onerror = (error) => { console.error('WebSocket 错误:', error); };
    }

    function startRealtime() {
        if (realtimeActive) return;
        if (ws && ws.readyState === WebSocket.OPEN) {
            ws.send(JSON.stringify({type: "start_realtime"}));
            realtimeActive = true;
            startStimuli();
            document.getElementById('btn-realtime-start').disabled = true;
            document.getElementById('btn-realtime-stop').disabled = false;
        } else {
            connectWebSocket();
            const checkInterval = setInterval(() => {
                if (ws && ws.readyState === WebSocket.OPEN) {
                    clearInterval(checkInterval);
                    ws.send(JSON.stringify({type: "start_realtime"}));
                    realtimeActive = true;
                    startStimuli();
                    document.getElementById('btn-realtime-start').disabled = true;
                    document.getElementById('btn-realtime-stop').disabled = false;
                }
            }, 200);
        }
    }

    function stopRealtime() {
        if (!realtimeActive) return;
        if (ws && ws.readyState === WebSocket.OPEN) ws.send(JSON.stringify({type: "stop_realtime"}));
        realtimeActive = false;
        document.getElementById('btn-realtime-start').disabled = false;
        document.getElementById('btn-realtime-stop').disabled = true;
    }

    function updateConfidenceBars(confidences) {
        const dirs = ['up', 'down', 'left', 'right'];
        for (let i=0; i<dirs.length; i++) {
            const bar = document.getElementById(`bar-${dirs[i]}`);
            const text = document.getElementById(`conf-${dirs[i]}`);
            if (bar) {
                const percent = Math.round(confidences[i] * 100);
                bar.style.width = percent + '%';
                if (text) text.innerText = percent + '%';
            }
        }
    }

    // ==================== 键盘控制 ====================
    function setupKeyboard() {
        document.addEventListener('keydown', (e) => {
            let cmd = null;
            const key = e.key;
            if (key === 'ArrowUp' || key === 'w' || key === 'W') cmd = 'up';
            else if (key === 'ArrowDown' || key === 's' || key === 'S') cmd = 'down';
            else if (key === 'ArrowLeft' || key === 'a' || key === 'A') cmd = 'left';
            else if (key === 'ArrowRight' || key === 'd' || key === 'D') cmd = 'right';
            if (cmd) {
                e.preventDefault();
                if (!evalMode) {
                    handleLocalMove(cmd);
                }
            }
        });
    }

    // ==================== 迷宫生成 ====================
    function bindMazeGenerate() {
        const genBtn = document.getElementById('generateMazeBtn');
        if (genBtn) {
            genBtn.addEventListener('click', () => {
                if (currentGame === 'maze' && activeGame instanceof MazeGame) {
                    const w = parseInt(document.getElementById('mazeWidth').value, 10);
                    const h = parseInt(document.getElementById('mazeHeight').value, 10);
                    activeGame.init(w, h);
                    activeGame.render(gameCtx);
                }
            });
        }
    }

    // ==================== 初始化 ====================
    function init() {
        activeGame = gameInstances.maze;
        activeGame.init();
        activeGame.render(gameCtx);
        setupKeyboard();
        bindMazeGenerate();
        connectWebSocket();

        document.getElementById('mode-offline').addEventListener('click', () => switchMode('offline'));
        document.getElementById('mode-online').addEventListener('click', () => switchMode('online'));

        document.getElementById('btn-maze').addEventListener('click', () => switchGame('maze'));
        document.getElementById('btn-snake').addEventListener('click', () => switchGame('snake'));
        document.getElementById('btn-racing').addEventListener('click', () => switchGame('racing'));

        document.getElementById('btn-maze-online').addEventListener('click', () => switchGameOnline('maze'));
        document.getElementById('btn-snake-online').addEventListener('click', () => switchGameOnline('snake'));
        document.getElementById('btn-racing-online').addEventListener('click', () => switchGameOnline('racing'));

        document.getElementById('btn-demo-start').addEventListener('click', startDemo);
        document.getElementById('btn-demo-stop').addEventListener('click', stopDemo);

        document.getElementById('btn-realtime-start').addEventListener('click', startRealtime);
        document.getElementById('btn-realtime-stop').addEventListener('click', stopRealtime);

        const evalStartBtn = document.getElementById('btn-eval-start');
if (evalStartBtn) evalStartBtn.addEventListener('click', () => {
    // 在线评测，来源为 online
    startEvalMode(null, 'online');
});

        ['demo-log', 'demo-summary', 'demo-progress'].forEach(id => {
            const el = document.getElementById(id);
            if (el) {
                el.style.userSelect = 'text';
                el.style.webkitUserSelect = 'text';
                el.style.mozUserSelect = 'text';
                el.style.msUserSelect = 'text';
                el.style.cursor = 'text';
            }
        });

        stimAnimationId = requestAnimationFrame(animateStim);
        stopStimuli();
    }

    window.stopDemo = stopDemo;
    window.startDemo = startDemo;
    window.switchGame = switchGame;
    window.startEvalMode = startEvalMode;
    window.stopEvalMode = stopEvalMode;

    init();
})();