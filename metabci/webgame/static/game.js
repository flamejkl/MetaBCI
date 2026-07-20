// game.js - 脑控游戏前端（双Canvas + 灰度刺激 + 丢帧监测 + 评测模式）
(function() {
    // ==================== 常量 ====================
    const WS_URL = `ws://${window.location.hostname}:8765`;
    const MAZE_DEFAULT_W = 31, MAZE_DEFAULT_H = 31;
    const DEMO_TIMEOUT_MS = 3000;
    const DEMO_MAX_RETRIES = 10;
    const WS_RECONNECT_DELAY = 3000;
    const N_CLASSES = 4;

    // Wolpaw ITR: B = log2(N)+P*log2(P)+(1-P)*log2((1-P)/(N-1)), ITR = B * 60/T
    function computeOnlineITR(steps) {
        const valid = steps.filter(s => s.match !== undefined && s.decision_time > 0);
        if (valid.length === 0) return null;
        const correct = valid.filter(s => s.match).length;
        const P = correct / valid.length;
        const avgT = valid.reduce((a, s) => a + s.decision_time, 0) / valid.length;
        if (P <= 1/N_CLASSES || avgT <= 0) return null;
        const B = Math.log2(N_CLASSES) + P*Math.log2(P) + (1-P)*Math.log2((1-P)/(N_CLASSES-1));
        const ITR = B * (60 / avgT);
        return { accuracy: P, avgTime: avgT, ITR: ITR, total: valid.length, correct };
    }

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
    let frameIntHead = 0;
    const FRAME_BUF_SIZE = 64;
    const frameIntervals = new Array(FRAME_BUF_SIZE).fill(0);
    let frameIntCount = 0;
    let frameCount = 0;           // 总帧数计数器

    // ==================== 刺激块布局 ====================
    // 离线实验参数 (run_ssvep_experiment.py: block=0.12屏宽, gap=0.1屏宽)
    // 全屏用 vw/vh 比例；普通模式用固定像素
    const STIM_RATIO = { block: 0.12, gap: 0.10 };
    const STIM_CONFIG = {
        blockWidth: 150,
        blockHeight: 168,
        gap: 65,
    };

    let positions = {};

    function initStimPositions() {
        const isFs = !!(document.fullscreenElement || document.webkitFullscreenElement);
        const cw = stimCanvas.width;
        const ch = stimCanvas.height;

        let bw, bh, gap;
        if (isFs) {
            // 匹配离线实验比例: block=15vw, gap=10vw
            bw = Math.round(STIM_RATIO.block * window.innerWidth);
            gap = Math.round(STIM_RATIO.gap * window.innerWidth);
            bh = Math.min(bw, ch - 8);   // 正方形，不超过 canvas 高度
            bw = Math.min(bw, bh);
        } else {
            bw = STIM_CONFIG.blockWidth;
            bh = STIM_CONFIG.blockHeight;
            gap = STIM_CONFIG.gap;
        }

        const totalW = bw * 4 + gap * 3;
        const startX = (cw - totalW) / 2;
        const baseY = (ch - bh) / 2;
        const dirOrder = ['up', 'down', 'left', 'right'];
        const pos = {};
        dirOrder.forEach((dir, i) => {
            const x = startX + i * (bw + gap) + bw / 2;
            const y = baseY + bh / 2;
            pos[dir] = { x, y, w: bw, h: bh };
        });
        return pos;
    }
    positions = initStimPositions();

    // ==================== 刺激绘制核心 ====================
    function drawStimuli(now) {
        // 采集模式非闪烁阶段：由专门函数绘制（不覆盖 index/rest 画面）
        if (collectMode && collectPhase !== 'stimulus') {
            if (collectPhase === 'preview') drawAllGray();
            else if (collectPhase === 'index' && collectTargetDir) drawIndexFrame(collectTargetDir);
            else if (collectPhase === 'rest') drawRestFrame();
            return;
        }

        stimCtx.clearRect(0, 0, stimCanvas.width, stimCanvas.height);

        const isFlashing = stimFlashing && stimStartTime !== null;
        const t = isFlashing ? (now - stimStartTime) / 1000 : 0;

        for (const dir of dirKeys) {
            let gray;
            if (isFlashing) {
                // 正弦波调制 — 与训练实验 stimtype='sinusoid' 完全一致
                const freq = stimFreqs[dir];
                const phase = stimPhases[dir] * Math.PI;
                const val = Math.sin(2 * Math.PI * freq * t + phase);
                gray = Math.floor(128 + 127 * val);   // 1~255 正弦灰度
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
    let fpsDisplay = 0;
    let fpsLastCheck = performance.now();
    let fpsFrameCount = 0;
    function animateStim(now) {
        frameCount++;
        // ---- FPS 计算（每秒更新） ----
        fpsFrameCount++;
        if (now - fpsLastCheck >= 1000) {
            fpsDisplay = Math.round(fpsFrameCount * 1000 / (now - fpsLastCheck));
            fpsFrameCount = 0;
            fpsLastCheck = now;
            const fpsEl = document.getElementById('fps-counter');
            const fpsFloat = document.getElementById('fps-counter-float');
            [fpsEl, fpsFloat].forEach(el => {
                if (el) {
                    el.textContent = fpsDisplay + ' FPS';
                    el.style.color = fpsDisplay >= 165 ? '#4caf50' : fpsDisplay >= 120 ? '#ffeb3b' : '#f44336';
                }
            });
        }
        if (stimFlashing) {
            const interval = now - lastStimFrameTime;
            lastStimFrameTime = now;
            frameIntervals[frameIntHead] = interval;
            frameIntHead = (frameIntHead + 1) % FRAME_BUF_SIZE;
            frameIntCount++;
            if (frameIntCount === 60) {
                let dropped = 0;
                const count = Math.min(frameIntCount, FRAME_BUF_SIZE);
                for (let i = 0; i < count; i++) {
                    if (frameIntervals[i] > 30) dropped++;
                }
                const dropRate = dropped / count;
                if (dropRate > 0.05) {
                    console.warn(`[刺激] 丢帧率 ${(dropRate*100).toFixed(1)}%`);
                }
                frameIntCount = 0;
                frameIntHead = 0;
            }
        }
        drawStimuli(now);
        // 全屏模式下每 30 帧更新一次分数显示
        if ((document.fullscreenElement || document.webkitFullscreenElement) && frameCount % 30 === 0) {
            const fsScore = document.getElementById('fs-score');
            const diaEl = document.getElementById('collectedDiamonds');
            const totEl = document.getElementById('totalDiamonds');
            const scoreEl = document.getElementById('score');
            if (fsScore && diaEl && totEl && scoreEl) {
                fsScore.textContent = `💎 ${diaEl.innerText}/${totEl.innerText}  🎯 ${scoreEl.innerText}分`;
            }
        }
        stimAnimationId = requestAnimationFrame(animateStim);
    }

    // ==================== 全屏切换 ====================
    function toggleFullscreen() {
        if (!document.fullscreenElement && !document.webkitFullscreenElement) {
            const el = document.documentElement;
            if (el.requestFullscreen) {
                el.requestFullscreen().catch(e => console.warn('全屏请求失败:', e));
            } else if (el.webkitRequestFullscreen) {
                el.webkitRequestFullscreen();
            }
        } else {
            if (document.exitFullscreen) {
                document.exitFullscreen();
            } else if (document.webkitExitFullscreen) {
                document.webkitExitFullscreen();
            }
        }
    }

    function applyFullscreenLayout(entering) {
        const fsHeader = document.querySelector('.fullscreen-header');
        const btn = document.getElementById('btn-fullscreen');
        const btnFs = document.getElementById('btn-fullscreen-fs');
        if (entering) {
            document.body.classList.add('fullscreen');
            if (fsHeader) fsHeader.style.display = 'flex';
            if (btn) btn.style.display = 'none';
            if (btnFs) btnFs.style.display = '';
            resizeGameForFullscreen();
        } else {
            document.body.classList.remove('fullscreen');
            if (fsHeader) fsHeader.style.display = 'none';
            if (btn) btn.style.display = '';
            if (btnFs) btnFs.style.display = 'none';
            gameCanvas.width = 800;
            gameCanvas.height = 800;
            gameCanvas.style.width = '800px';
            gameCanvas.style.height = '800px';
            gameCanvas.style.borderRadius = '20px 20px 0 0';
            if (activeGame) activeGame.render(gameCtx);
        }
        updateStimLayout();
        // 确保动画循环运行（全屏切换可能导致 rAF 暂停）
        if (!stimAnimationId) {
            stimAnimationId = requestAnimationFrame(animateStim);
        }
    }

    function updateStimLayout() {
        const isFs = !!(document.fullscreenElement || document.webkitFullscreenElement);
        if (isFs) {
            stimCanvas.width = window.innerWidth;
            stimCanvas.height = Math.round(window.innerHeight * 0.26);
            stimCanvas.style.width = '100vw';
            stimCanvas.style.height = '26vh';
        } else {
            stimCanvas.width = 800;
            stimCanvas.height = 180;
            stimCanvas.style.width = '800px';
            stimCanvas.style.height = '180px';
        }
        positions = initStimPositions();
        drawStimuli(performance.now());
    }

    function resizeGameForFullscreen() {
        const headerH = 36;
        const stimH = Math.round(window.innerHeight * 0.22);
        const availH = window.innerHeight - headerH - stimH;
        const availW = window.innerWidth;
        // 正方形游戏画布，取最大内接正方形，留 5% 边距
        const size = Math.floor(Math.min(availH * 0.92, availW * 0.90));
        gameCanvas.width = size;
        gameCanvas.height = size;
        gameCanvas.style.width = size + 'px';
        gameCanvas.style.height = size + 'px';
        gameCanvas.style.borderRadius = '8px';
        // 重初始化游戏以适配新 canvas 尺寸
        if (activeGame) {
            activeGame.init();
            activeGame.render(gameCtx);
        }
        // 更新全屏标题栏得分
        const fsScore = document.getElementById('fs-score');
        if (fsScore) {
            const scoreEl = document.getElementById('score');
            const diaEl = document.getElementById('collectedDiamonds');
            const totEl = document.getElementById('totalDiamonds');
            if (scoreEl && diaEl && totEl) {
                fsScore.textContent = `💎 ${diaEl.innerText}/${totEl.innerText}  🎯 ${scoreEl.innerText}分`;
            }
        }
    }

    function onFullscreenChange() {
        const isFs = !!(document.fullscreenElement || document.webkitFullscreenElement);
        applyFullscreenLayout(isFs);
    }

    function onFullscreenResize() {
        if (document.fullscreenElement || document.webkitFullscreenElement) {
            resizeGameForFullscreen();
            updateStimLayout();
        }
    }

    // ==================== 对外控制接口 ====================
    function startStimuli() {
        if (!stimFlashing) {
            stimFlashing = true;
            stimStartTime = performance.now();
            lastStimFrameTime = stimStartTime;
            frameIntHead = 0;
            frameIntCount = 0;
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
        stop() {}  // cleanup timers/loops
    }

    // ==================== 迷宫游戏（视觉优化版） ====================
    class MazeGame extends BaseGame {
        constructor() {
            super();
            this.state = null;
            this._animId = null;       // rAF id for animations
            this._animFrom = null;     // [r, c] animation start
            this._animTo = null;       // [r, c] animation end
            this._animStart = 0;       // timestamp
            this._animDur = 160;       // ms for smooth player slide
            this._particles = [];      // {x, y, vx, vy, life, maxLife, color}
            this._lastFrame = 0;
            this._goalGlow = 0;        // goal pulse phase
            this._lastFaceDir = 'down'; // remember last facing direction
        }

        init(width = MAZE_DEFAULT_W, height = MAZE_DEFAULT_H) {
            this._stopAnim();
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
                cell_size: cellSize, diamonds, collectedDiamonds: 0, totalDiamonds: dCount,
                won: false
            };
            this._animFrom = null; this._animTo = null;
            this._particles = [];
            this._lastFrame = performance.now();
            this._startAnimLoop();
            this._updateUI();
        }

        stop() { this._stopAnim(); }

        // ================ animation loop ================
        _startAnimLoop() {
            const loop = (now) => {
                const dt = now - this._lastFrame;
                this._lastFrame = now;
                this._goalGlow += dt * 0.003;  // slow pulse
                // update particles
                this._particles = this._particles.filter(p => {
                    p.x += p.vx * dt / 1000;
                    p.y += p.vy * dt / 1000;
                    p.life -= dt;
                    return p.life > 0;
                });
                this.render(gameCtx);
                // keep loop alive if animating or particles active
                if (this._animTo || this._particles.length > 0 || this.state?.won) {
                    this._animId = requestAnimationFrame(loop);
                } else {
                    this._animId = null;
                }
            };
            if (!this._animId) this._animId = requestAnimationFrame(loop);
        }
        _stopAnim() {
            if (this._animId) { cancelAnimationFrame(this._animId); this._animId = null; }
        }
        _wakeAnim() {
            if (!this._animId && this.state) {
                this._lastFrame = performance.now();
                this._animId = requestAnimationFrame((now) => {
                    this._lastFrame = now;
                    const loop = (now2) => {
                        const dt = now2 - this._lastFrame;
                        this._lastFrame = now2;
                        this._goalGlow += dt * 0.003;
                        this._particles = this._particles.filter(p => {
                            p.x += p.vx * dt / 1000;
                            p.y += p.vy * dt / 1000;
                            p.life -= dt;
                            return p.life > 0;
                        });
                        this.render(gameCtx);
                        if (this._animTo || this._particles.length > 0 || this.state?.won) {
                            this._animId = requestAnimationFrame(loop);
                        } else {
                            this._animId = null;
                        }
                    };
                    this._animId = requestAnimationFrame(loop);
                });
            }
        }

        // ================ spawn particles ================
        _burst(x, y, color, count) {
            for (let i = 0; i < count; i++) {
                const angle = Math.random() * Math.PI * 2;
                const speed = 40 + Math.random() * 120;
                this._particles.push({
                    x, y,
                    vx: Math.cos(angle) * speed,
                    vy: Math.sin(angle) * speed - 30,
                    life: 300 + Math.random() * 400,
                    maxLife: 700,
                    color: color
                });
            }
        }

        // ================ maze generation (unchanged) ================
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
                    if (nx>0 && nx<width-1 && ny>0 && ny<height-1 && maze[ny][nx]===1)
                        walls.push({ wx: x+dx/2, wy: y+dy/2, nx, ny });
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

        // ================ brain input ================
        handleMove(cmd) {
            const s = this.state;
            if (!s || s.won) return false;
            let newPos = [...s.player];
            switch(cmd) {
                case 'up': newPos[0]--; break;
                case 'down': newPos[0]++; break;
                case 'left': newPos[1]--; break;
                case 'right': newPos[1]++; break;
                default: return false;
            }
            if (newPos[0] < 0 || newPos[0] >= s.maze.length ||
                newPos[1] < 0 || newPos[1] >= s.maze[0].length) return false;
            if (s.maze[newPos[0]][newPos[1]] !== 0) return false;

            // Start smooth-move animation from current to new cell
            this._animFrom = [...s.player];
            this._animTo = newPos;
            this._animStart = performance.now();
            this._wakeAnim();

            s.player = newPos;
            s.moves++;

            // Diamond collect?
            const idx = s.diamonds.findIndex(d => d[0]===newPos[0] && d[1]===newPos[1]);
            if (idx !== -1) {
                s.diamonds.splice(idx,1);
                s.collectedDiamonds++;
                s.score += 10;
                // compute world position for particle burst
                const cs = s.cell_size;
                const offX = (gameCanvas.width - s.maze[0].length*cs)/2;
                const offY = (gameCanvas.height - s.maze.length*cs)/2;
                const wx = offX + newPos[1]*cs + cs/2;
                const wy = offY + newPos[0]*cs + cs/2;
                this._burst(wx, wy, '#ffcc00', 12);
                this._updateUI();
            }

            // Goal reached?
            if (newPos[0]===s.goal[0] && newPos[1]===s.goal[1]) {
                const rate = s.totalDiamonds > 0 ? (s.collectedDiamonds/s.totalDiamonds)*100 : 0;
                let bonus = 10, msg = '收集率不足50%';
                if (rate===100) { bonus=50; msg='✨ 完美收集！'; }
                else if (rate>=80) { bonus=30; msg='收集率超过80%'; }
                else if (rate>=50) { bonus=20; msg='收集率超过50%'; }
                s.score += bonus;
                s.won = true;
                s._winMsg = msg;
                s._winBonus = bonus;
                s._winRate = rate;
                // goal burst
                const cs = s.cell_size;
                const offX = (gameCanvas.width - s.maze[0].length*cs)/2;
                const offY = (gameCanvas.height - s.maze.length*cs)/2;
                this._burst(offX + newPos[1]*cs + cs/2, offY + newPos[0]*cs + cs/2, '#ffaa00', 30);
                this._updateUI();
                this._wakeAnim();
                // Auto-handle demo/eval mode after a short delay
                if (evalMode) {
                    setTimeout(() => stopEvalMode(), 1500);
                } else if (demoActive) {
                    setTimeout(() => stopDemo(), 1500);
                }
            }
            this._updateUI();
            return true;
        }

        // ================ rendering ================
        render(ctx) {
            const s = this.state;
            if (!s) return;
            const cs = s.cell_size;
            const rows = s.maze.length, cols = s.maze[0].length;
            const offX = (gameCanvas.width - cols*cs)/2;
            const offY = (gameCanvas.height - rows*cs)/2;

            ctx.clearRect(0, 0, gameCanvas.width, gameCanvas.height);

            // ---- background ----
            ctx.fillStyle = '#0d1117';
            ctx.fillRect(0, 0, gameCanvas.width, gameCanvas.height);

            // ---- cells ----
            for (let i = 0; i < rows; i++) {
                for (let j = 0; j < cols; j++) {
                    const x = offX + j*cs, y = offY + i*cs;
                    if (s.maze[i][j] === 1) {
                        // Wall: dark gradient with slight 3D bevel
                        const grad = ctx.createLinearGradient(x, y, x + cs, y + cs);
                        grad.addColorStop(0, '#1e2430');
                        grad.addColorStop(0.4, '#2a3040');
                        grad.addColorStop(0.6, '#252c38');
                        grad.addColorStop(1, '#1a1f28');
                        ctx.fillStyle = grad;
                        ctx.fillRect(x, y, cs, cs);
                        // top-left highlight edge
                        ctx.strokeStyle = 'rgba(255,255,255,0.06)';
                        ctx.lineWidth = 1;
                        ctx.beginPath(); ctx.moveTo(x, y+cs); ctx.lineTo(x, y); ctx.lineTo(x+cs, y); ctx.stroke();
                        // bottom-right shadow edge
                        ctx.strokeStyle = 'rgba(0,0,0,0.3)';
                        ctx.beginPath(); ctx.moveTo(x+cs, y); ctx.lineTo(x+cs, y+cs); ctx.lineTo(x, y+cs); ctx.stroke();
                    } else {
                        // Path: warm sandstone
                        ctx.fillStyle = '#c8b89a';
                        ctx.fillRect(x, y, cs, cs);
                        // subtle pattern
                        ctx.fillStyle = 'rgba(0,0,0,0.03)';
                        if ((i+j)%2===0) ctx.fillRect(x, y, cs, cs);
                    }
                }
            }

            // ---- diamonds ----
            const now = performance.now();
            for (let d of s.diamonds) {
                const cx = offX + d[1]*cs + cs/2, cy = offY + d[0]*cs + cs/2;
                const r = cs * 0.32;
                const sparkle = 0.7 + 0.3 * Math.sin(now*0.005 + d[0]*d[1]);
                // diamond shape
                ctx.save();
                ctx.translate(cx, cy);
                ctx.rotate(Math.PI/4);
                ctx.fillStyle = `rgba(255,200,50,${sparkle})`;
                ctx.shadowColor = '#ffcc00';
                ctx.shadowBlur = 6 * sparkle;
                ctx.beginPath();
                ctx.moveTo(0, -r);
                ctx.lineTo(r*0.6, 0);
                ctx.lineTo(0, r);
                ctx.lineTo(-r*0.6, 0);
                ctx.closePath();
                ctx.fill();
                // inner highlight
                ctx.fillStyle = `rgba(255,255,200,${sparkle*0.7})`;
                ctx.beginPath();
                ctx.moveTo(0, -r*0.45);
                ctx.lineTo(r*0.25, 0);
                ctx.lineTo(0, r*0.45);
                ctx.lineTo(-r*0.25, 0);
                ctx.closePath();
                ctx.fill();
                ctx.shadowBlur = 0;
                ctx.restore();
            }

            // ---- goal portal ----
            const gx = offX + s.goal[1]*cs + cs/2;
            const gy = offY + s.goal[0]*cs + cs/2;
            const glow = 0.5 + 0.5 * Math.sin(this._goalGlow);
            // outer glow ring
            for (let ring = 3; ring >= 0; ring--) {
                const rr = cs * (0.25 + ring * 0.12) * (1 + 0.08 * glow);
                const alpha = 0.15 - ring * 0.03;
                ctx.fillStyle = `rgba(255,180,50,${alpha})`;
                ctx.beginPath(); ctx.arc(gx, gy, rr, 0, Math.PI*2); ctx.fill();
            }
            // star icon
            ctx.fillStyle = '#fff';
            ctx.font = `${cs*0.55}px "Segoe UI"`;
            ctx.textAlign = 'center'; ctx.textBaseline = 'middle';
            ctx.shadowColor = '#ffaa00';
            ctx.shadowBlur = 10 + 5 * glow;
            ctx.fillText('⭐', gx, gy);
            ctx.shadowBlur = 0;
            ctx.textAlign = 'start'; ctx.textBaseline = 'alphabetic';

            // ---- player ----
            let pr, pc;
            if (this._animTo) {
                // Interpolate between _animFrom and _animTo
                const elapsed = now - this._animStart;
                const t = Math.min(elapsed / this._animDur, 1.0);
                const ease = t < 0.5 ? 2*t*t : -1+(4-2*t)*t;  // easeInOutQuad
                pr = this._animFrom[0] + (this._animTo[0] - this._animFrom[0]) * ease;
                pc = this._animFrom[1] + (this._animTo[1] - this._animFrom[1]) * ease;
                if (t >= 1.0) {
                    this._animFrom = null; this._animTo = null;
                    if (s.won) {
                        // show victory overlay, keep rendering
                    } else if (this._particles.length === 0) {
                        // render one final frame then stop loop
                    }
                }
            } else {
                pr = s.player[0]; pc = s.player[1];
            }
            const px = offX + pc*cs + cs/2;
            const py = offY + pr*cs + cs/2;
            const scale = cs / 32;
            // Walk animation: 3-frame cycle (450ms), dramatic stride
            const bob = this._animTo ? Math.abs(Math.sin((now - this._animStart) / this._animDur * Math.PI)) * cs * 0.1 : 0;
            const charY = py + bob;

            let faceDir = this._lastFaceDir || 'down';
            if (this._animTo) {
                const dr = this._animTo[0] - this._animFrom[0];
                const dc = this._animTo[1] - this._animFrom[1];
                if (dr < 0) faceDir = 'up';
                else if (dr > 0) faceDir = 'down';
                else if (dc < 0) faceDir = 'left';
                else if (dc > 0) faceDir = 'right';
            }
            const walking = !!this._animTo;
            const cycleMs = 450;
            const walkT = walking ? ((now - this._animStart) % cycleMs) / cycleMs : 0;
            // 3-frame cycle: 0=left stride, 1=passing, 2=right stride
            let walkFrame;
            if (walkT < 0.33) walkFrame = 0;
            else if (walkT < 0.66) walkFrame = 1;
            else walkFrame = 2;
            const stride = walking ? 3.5 * scale : 0;  // leg separation in pixels

            this._drawKnight(ctx, px, charY, cs, scale, faceDir, walkFrame, stride, walkT);
            if (walking) this._lastFaceDir = faceDir;

            // ---- particles ----
            for (let p of this._particles) {
                const alpha = Math.max(0, p.life / p.maxLife);
                const size = 2 + 4 * alpha;
                ctx.fillStyle = p.color.replace(')', `,${alpha})`).replace('rgb', 'rgba');
                if (p.color.startsWith('#')) {
                    ctx.globalAlpha = alpha;
                    ctx.fillStyle = p.color;
                }
                ctx.beginPath(); ctx.arc(p.x, p.y, size, 0, Math.PI*2); ctx.fill();
            }
            ctx.globalAlpha = 1;

            // ---- HUD (bottom-left overlay) ----
            this._drawHUD(ctx, s, offX, offY, rows, cs);

            // ---- victory overlay ----
            if (s.won) {
                this._drawVictory(ctx, s);
            }
        }

        // ===== detailed knight: 4-direction + 3-frame walk + layered armor =====
        _drawKnight(ctx, cx, cy, cs, s, dir, frame, stride, walkT) {
            ctx.save();
            ctx.translate(cx, cy);

            // --------------- palette ---------------
            const M  = '#d0d0d8';   // armor mid
            const ML = '#e8e8f0';   // armor highlight
            const MD = '#8a8a96';   // armor shadow
            const G  = '#daa520';   // gold trim
            const GD = '#9a7510';   // gold dark
            const R  = '#cc2020';   // plume / cape red
            const RD = '#881818';   // dark red
            const B  = '#332211';   // dark brown (boots, grip)
            const SK = '#e0c098';   // skin
            const eyeW = '#ffffff'; // eye whites

            // --------------- body layout ---------------
            const headR  = cs * 0.18;
            const headY  = -cs * 0.30;        // helmet centre Y
            const bodyTop = headY + headR * 0.85;
            const legTop  = cs * 0.18;
            const feetBot = cs * 0.42;
            const bodyH   = legTop - bodyTop;
            const legLen  = feetBot - legTop;
            const legW    = 3.5 * s;
            const isSide  = (dir === 'left' || dir === 'right');
            const sg      = dir === 'right' ? 1 : -1;  // side sign

            // walk stride
            const st0 = frame === 0 ? stride : (frame === 2 ? -stride : 0);
            const st1 = frame === 2 ? stride : (frame === 0 ? -stride : 0);

            // ---- ground shadow ----
            const shY = feetBot + 1*s;
            ctx.fillStyle = 'rgba(0,0,0,0.22)';
            ctx.beginPath(); ctx.ellipse(0, shY, cs*0.24, cs*0.06, 0, 0, Math.PI*2); ctx.fill();

            // ================================================================
            //  CAPE  (behind everything except back view where it's prominent)
            // ================================================================
            if (dir === 'up') {
                ctx.fillStyle = R;
                ctx.beginPath();
                ctx.moveTo(-7*s, bodyTop + 6*s);
                ctx.lineTo(7*s, bodyTop + 6*s);
                ctx.quadraticCurveTo(12*s, legTop, 9*s, feetBot + 4*s);
                ctx.lineTo(-9*s, feetBot + 4*s);
                ctx.quadraticCurveTo(-12*s, legTop, -7*s, bodyTop + 6*s);
                ctx.fill();
                // cape fold lines
                ctx.strokeStyle = RD; ctx.lineWidth = 0.5;
                ctx.beginPath(); ctx.moveTo(-2*s, bodyTop+10*s); ctx.lineTo(-4*s, feetBot); ctx.stroke();
                ctx.beginPath(); ctx.moveTo(2*s, bodyTop+10*s); ctx.lineTo(4*s, feetBot); ctx.stroke();
            } else if (isSide) {
                // Small cape visible behind shoulders
                ctx.fillStyle = R;
                ctx.beginPath();
                ctx.moveTo(-3*s, bodyTop + 3*s);
                ctx.quadraticCurveTo(-5*s - sg*2*s, bodyTop + 8*s, -3*s - sg*3*s, legTop + 2*s);
                ctx.lineTo(2*s - sg*2*s, legTop + 2*s);
                ctx.quadraticCurveTo(0, bodyTop + 6*s, 3*s, bodyTop + 3*s);
                ctx.fill();
            }

            // ================================================================
            //  LEGS — greaves + poleyns (knee guards) + sabatons (armored boots)
            // ================================================================
            const drawLegFrontBack = (lx, off) => {
                const y0 = legTop + off, y1 = y0 + legLen - Math.abs(off)*0.3;
                // greave (shin guard)
                ctx.fillStyle = M;
                ctx.fillRect(lx, y0, legW, y1 - y0);
                // greave highlight
                ctx.fillStyle = ML;
                ctx.fillRect(lx + legW*0.2, y0, legW*0.3, y1 - y0);
                // poleyn (knee guard — round)
                ctx.fillStyle = M;
                ctx.beginPath();
                ctx.arc(lx + legW/2, y0 + 1*s, legW*0.8, 0, Math.PI*2);
                ctx.fill();
                ctx.fillStyle = G;
                ctx.beginPath();
                ctx.arc(lx + legW/2, y0 + 1*s, legW*0.35, 0, Math.PI*2);
                ctx.fill();
                // sabaton (armored boot)
                ctx.fillStyle = MD;
                ctx.fillRect(lx - 1.5*s, y1 - 3*s, legW + 3*s, 4*s);
                ctx.fillStyle = B;
                ctx.fillRect(lx - 1*s, y1 - 1*s, legW + 2*s, 1.5*s);
            };
            if (isSide) {
                const bx = -1*s;
                // far leg
                ctx.fillStyle = MD;
                ctx.fillRect(bx - 1*s, legTop + st1*0.3, legW*0.7, legLen);
                // front leg
                const fx = bx + 5*s + st0*0.5*sg;
                ctx.fillStyle = M;
                ctx.fillRect(fx, legTop, legW*0.9, legLen);
                ctx.fillStyle = ML;
                ctx.fillRect(fx + legW*0.2, legTop, legW*0.3, legLen);
                // knee
                ctx.fillStyle = M;
                ctx.beginPath(); ctx.arc(fx + legW*0.45, legTop + 1.5*s, legW*0.75, 0, Math.PI*2); ctx.fill();
                ctx.fillStyle = G;
                ctx.beginPath(); ctx.arc(fx + legW*0.45, legTop + 1.5*s, legW*0.3, 0, Math.PI*2); ctx.fill();
                // boot
                ctx.fillStyle = B;
                ctx.fillRect(fx - 2*s, legTop + legLen - 3*s, legW*0.9 + 4*s, 3.5*s);
            } else {
                drawLegFrontBack(-legW - 2.5*s, st0 * 0.5);
                drawLegFrontBack(2.5*s, st1 * 0.5);
            }

            // ================================================================
            //  TORSO — layered plates + fauld (armored skirt) + belt
            // ================================================================
            if (isSide) {
                const bw = 5*s;
                // breastplate body
                ctx.fillStyle = M;
                ctx.fillRect(-bw + 2*s, bodyTop, bw*2 - 2*s, bodyH);
                // highlight curve
                const grad = ctx.createLinearGradient(0, bodyTop, 0, legTop);
                grad.addColorStop(0, ML); grad.addColorStop(0.5, M); grad.addColorStop(1, MD);
                ctx.fillStyle = grad;
                ctx.fillRect(-bw + 3*s, bodyTop, bw - 1*s, bodyH);
                // fauld plates (horizontal strips at waist)
                for (let j = 0; j < 3; j++) {
                    ctx.fillStyle = j % 2 === 0 ? M : MD;
                    ctx.fillRect(-bw + 1.5*s, legTop - 9*s + j*3*s, bw*2 - 3*s, 2.8*s);
                }
                // belt
                ctx.fillStyle = G;
                ctx.fillRect(-bw + 1*s, legTop - 4*s, bw*2 - 2*s, 2*s);
            } else if (dir === 'up') {
                const bwt = 6.5*s, bwb = 8.5*s;
                // backplate
                ctx.fillStyle = MD;
                ctx.beginPath();
                ctx.moveTo(-bwt, bodyTop); ctx.lineTo(bwt, bodyTop);
                ctx.lineTo(bwb, legTop); ctx.lineTo(-bwb, legTop);
                ctx.closePath(); ctx.fill();
                // shoulder blades (two dark ovals)
                ctx.fillStyle = '#707078';
                ctx.beginPath(); ctx.ellipse(-4*s, bodyTop + 4*s, 3*s, 4*s, 0.2, 0, Math.PI*2); ctx.fill();
                ctx.beginPath(); ctx.ellipse(4*s, bodyTop + 4*s, 3*s, 4*s, -0.2, 0, Math.PI*2); ctx.fill();
            } else {
                // FRONT — detailed breastplate
                const bwt = 9*s, bwb = 11*s;
                // main chest
                const chestGrad = ctx.createLinearGradient(0, bodyTop, 0, legTop);
                chestGrad.addColorStop(0, ML); chestGrad.addColorStop(0.3, M); chestGrad.addColorStop(0.7, M); chestGrad.addColorStop(1, MD);
                ctx.fillStyle = chestGrad;
                ctx.beginPath();
                ctx.moveTo(-bwt, bodyTop); ctx.lineTo(bwt, bodyTop);
                ctx.lineTo(bwb, legTop); ctx.lineTo(-bwb, legTop);
                ctx.closePath(); ctx.fill();
                // pectoral definition (two curved lines)
                ctx.strokeStyle = MD; ctx.lineWidth = 0.7;
                ctx.beginPath();
                ctx.moveTo(-5*s, bodyTop + 4*s); ctx.quadraticCurveTo(0, bodyTop + 1*s, 5*s, bodyTop + 4*s);
                ctx.stroke();
                // center ridge highlight
                ctx.fillStyle = ML;
                ctx.fillRect(-2*s, bodyTop + 3*s, 4*s, bodyH - 3*s);
                // fauld (3 overlapping plates at waist)
                for (let j = 0; j < 3; j++) {
                    const py = legTop - 10*s + j*3.3*s;
                    const pw = bwb - 2*s + j*1.5*s;
                    ctx.fillStyle = j % 2 === 0 ? M : MD;
                    ctx.fillRect(-pw, py, pw*2, 3*s);
                    if (j % 2 === 0) {
                        ctx.fillStyle = G;
                        ctx.fillRect(-pw, py, pw*2, 0.6*s);
                    }
                }
                // belt + buckle
                ctx.fillStyle = G;
                ctx.fillRect(-bwb + 3*s, legTop - 3.5*s, (bwb - 3*s)*2, 3*s);
                ctx.fillStyle = GD;
                ctx.fillRect(-2.5*s, legTop - 5*s, 5*s, 5*s);  // buckle
                ctx.fillStyle = G;
                ctx.fillRect(-1.5*s, legTop - 4*s, 3*s, 3*s);
            }

            // ================================================================
            //  PAULDRONS (shoulder armor) — prominent rounded shapes
            // ================================================================
            if (isSide) {
                // Near shoulder
                ctx.fillStyle = M;
                ctx.beginPath(); ctx.arc(5*s, bodyTop + 2*s, 5*s, 0, Math.PI*2); ctx.fill();
                ctx.fillStyle = ML;
                ctx.beginPath(); ctx.arc(5*s, bodyTop + 1*s, 3*s, 0, Math.PI*2); ctx.fill();
                ctx.fillStyle = G;
                ctx.beginPath(); ctx.arc(6*s, bodyTop + 0.5*s, 1.5*s, 0, Math.PI*2); ctx.fill();  // rivet
            } else if (dir === 'down') {
                for (let side = -1; side <= 1; side += 2) {
                    const sx = side * 8.5*s;
                    ctx.fillStyle = M;
                    ctx.beginPath(); ctx.arc(sx, bodyTop + 2*s, 5.5*s, 0, Math.PI*2); ctx.fill();
                    ctx.fillStyle = ML;
                    ctx.beginPath(); ctx.arc(sx, bodyTop + 1*s, 3.5*s, 0, Math.PI*2); ctx.fill();
                    // gold trim rim
                    ctx.strokeStyle = G; ctx.lineWidth = 1*s;
                    ctx.beginPath(); ctx.arc(sx, bodyTop + 2*s, 5*s, Math.PI*0.4, Math.PI*1.6); ctx.stroke();
                    ctx.lineWidth = 1;
                    ctx.fillStyle = G;
                    ctx.beginPath(); ctx.arc(sx + side*1*s, bodyTop + 0.5*s, 1.5*s, 0, Math.PI*2); ctx.fill();
                }
            } else {
                // back view: smaller pauldrons
                for (let side = -1; side <= 1; side += 2) {
                    ctx.fillStyle = MD;
                    ctx.beginPath(); ctx.arc(side * 8*s, bodyTop + 2*s, 4.5*s, 0, Math.PI*2); ctx.fill();
                }
            }

            // ================================================================
            //  ARMS  — vambraces + gauntlets + sword
            // ================================================================
            const armW = 2.8*s, armLen = 9*s;
            if (isSide) {
                const ax0 = 2*s;
                const ay0 = bodyTop + 3*s;
                const sw = frame === 0 ? st0*0.3*sg : (frame === 2 ? st1*0.3*sg : 0);
                // vambrace (forearm armor)
                ctx.fillStyle = M;
                ctx.fillRect(ax0 + sw - 1.5*s, ay0, armW, armLen);
                ctx.fillStyle = ML;
                ctx.fillRect(ax0 + sw - 0.5*s, ay0, armW*0.3, armLen);
                // gauntlet
                ctx.fillStyle = MD;
                ctx.fillRect(ax0 + sw - 2*s, ay0 + armLen - 2*s, armW + 1.5*s, 3*s);
                // === SWORD (horizontal, pointing forward) ===
                const swX = ax0 + sw + armW;
                const swY = ay0 + armLen*0.45;
                // crossguard
                ctx.fillStyle = G;
                ctx.fillRect(swX - 1*s, swY - 3*s, 3*s, 6*s);
                // grip
                ctx.fillStyle = B;
                ctx.fillRect(swX + 2*s, swY - 2*s, 4*s, 4*s);
                // blade (long, pointed)
                ctx.fillStyle = '#f0f0f8';
                ctx.fillRect(swX + 6*s, swY - 1.5*s, 10*s, 3*s);
                ctx.fillStyle = ML;
                ctx.fillRect(swX + 6*s, swY - 1*s, 10*s, 1.2*s);  // blade highlight
                // blade tip
                ctx.beginPath();
                ctx.moveTo(swX + 16*s, swY);
                ctx.lineTo(swX + 13*s, swY - 2.5*s);
                ctx.lineTo(swX + 13*s, swY + 2.5*s);
                ctx.closePath(); ctx.fill();
                // pommel
                ctx.fillStyle = G;
                ctx.beginPath(); ctx.arc(swX + 2*s, swY, 2*s, 0, Math.PI*2); ctx.fill();
            } else if (dir === 'down') {
                // Left arm
                ctx.fillStyle = M;
                ctx.fillRect(-12*s + st1*0.2, bodyTop + 3*s, armW, armLen);
                ctx.fillStyle = ML;
                ctx.fillRect(-11.5*s + st1*0.2, bodyTop + 3*s, armW*0.3, armLen);
                ctx.fillStyle = MD;
                ctx.fillRect(-12.5*s + st1*0.2, bodyTop + 3*s + armLen - 2*s, armW + 1.5*s, 3*s);
                // Right arm + sword
                const rx = 9*s + st0*0.2, ry = bodyTop + 3*s;
                ctx.fillStyle = M;
                ctx.fillRect(rx, ry, armW, armLen);
                ctx.fillStyle = ML;
                ctx.fillRect(rx + armW*0.2, ry, armW*0.3, armLen);
                ctx.fillStyle = MD;
                ctx.fillRect(rx - 0.5*s, ry + armLen - 2*s, armW + 1.5*s, 3*s);
                // Sword
                ctx.save();
                ctx.translate(rx + armW/2, ry + armLen);
                ctx.rotate(0.25);
                // crossguard
                ctx.fillStyle = G;
                ctx.fillRect(-4*s, -1.5*s, 8*s, 3*s);
                // grip
                ctx.fillStyle = B;
                ctx.fillRect(-1*s, -0.5*s, 2.5*s, 5*s);
                // pommel
                ctx.fillStyle = G;
                ctx.beginPath(); ctx.arc(0, 5*s, 2*s, 0, Math.PI*2); ctx.fill();
                // blade
                ctx.fillStyle = '#f0f0f8';
                ctx.fillRect(-1.2*s, -14*s, 2.4*s, 13*s);
                ctx.fillStyle = ML;
                ctx.fillRect(-0.5*s, -13*s, 0.8*s, 11*s);
                ctx.restore();
            } else {
                // back: arms at sides, sword scabbard on back
                ctx.fillStyle = MD;
                ctx.fillRect(-12*s, bodyTop + 3*s, armW, armLen*0.8);
                ctx.fillRect(9.5*s, bodyTop + 3*s, armW, armLen*0.8);
                // Scabbard diagonal across back
                ctx.fillStyle = G;
                ctx.save();
                ctx.translate(2*s, headY + headR);
                ctx.rotate(-0.2);
                ctx.fillRect(-2*s, 0, 4*s, 14*s);
                ctx.fillStyle = B;
                ctx.fillRect(0, 4*s, 2*s, 4*s);  // grip end
                ctx.restore();
            }

            // ================================================================
            //  HELMET — great helm style with十字 visor + breathing holes + crest
            // ================================================================
            if (isSide) {
                // Helmet dome
                ctx.fillStyle = M;
                ctx.beginPath(); ctx.arc(0, headY, headR, 0, Math.PI*2); ctx.fill();
                ctx.fillStyle = ML;
                ctx.beginPath(); ctx.arc(0, headY - headR*0.2, headR*0.7, 0, Math.PI*2); ctx.fill();
                // Visor — prominent dark bar
                const vx = sg * headR*0.15, vy = headY - 0.5*s;
                ctx.fillStyle = '#222';
                ctx.fillRect(vx, vy, headR*0.55, 3*s);
                // Breathing holes (tiny dots on lower front)
                ctx.fillStyle = '#111';
                for (let h = 0; h < 3; h++) {
                    ctx.beginPath();
                    ctx.arc(sg * headR*0.5, vy + 4*s + h*1.8*s, 0.7*s, 0, Math.PI*2);
                    ctx.fill();
                }
                // Neck guard
                ctx.fillStyle = MD;
                ctx.fillRect(-3*s, headY + headR*0.7, 6*s, 3*s);
                // Crest/plume
                ctx.fillStyle = R;
                ctx.beginPath();
                ctx.moveTo(0, headY - headR*0.95);
                ctx.quadraticCurveTo(-sg*headR*1.2, headY - headR*1.6, -sg*headR*0.5, headY - headR*1.3);
                ctx.quadraticCurveTo(sg*headR*0.1, headY - headR*1.0, 0, headY - headR*0.8);
                ctx.fill();
            } else if (dir === 'up') {
                // Full helmet back
                ctx.fillStyle = MD;
                ctx.beginPath(); ctx.arc(0, headY, headR*1.08, 0, Math.PI*2); ctx.fill();
                // Neck guard (wider at back)
                ctx.fillStyle = MD;
                ctx.fillRect(-4.5*s, headY + headR*0.7, 9*s, 3*s);
                // Plume
                ctx.fillStyle = R;
                ctx.beginPath();
                ctx.moveTo(0, headY - headR*0.9);
                ctx.quadraticCurveTo(4*s, headY - headR*1.5, 6*s, headY - headR*0.9);
                ctx.quadraticCurveTo(2*s, headY - headR*0.7, 0, headY - headR*0.5);
                ctx.fill();
            } else {
                // FRONT — great helm
                // Dome
                const helmGrad = ctx.createRadialGradient(0, headY - headR*0.3, headR*0.1, 0, headY, headR*1.05);
                helmGrad.addColorStop(0, ML); helmGrad.addColorStop(0.6, M); helmGrad.addColorStop(1, MD);
                ctx.fillStyle = helmGrad;
                ctx.beginPath(); ctx.arc(0, headY, headR, 0, Math.PI*2); ctx.fill();
                // 十字 visor (iconic crusader-style cross slit)
                ctx.fillStyle = '#1a1a1a';
                const visW = headR * 0.55, visH = headR * 0.15;
                // horizontal slit
                ctx.fillRect(-visW, headY - 0.5*s, visW*2, 2.8*s);
                // vertical slit (T-shape)
                ctx.fillRect(-1.2*s, headY - 1*s, 2.4*s, 5*s);
                // Breathing holes (3 each side of vertical slit)
                ctx.fillStyle = '#111';
                for (let side = -1; side <= 1; side += 2) {
                    for (let h = 0; h < 3; h++) {
                        ctx.beginPath();
                        ctx.arc(side * headR*0.35, headY + 3*s + h*1.8*s, 0.6*s, 0, Math.PI*2);
                        ctx.fill();
                    }
                }
                // Neck guard
                ctx.fillStyle = MD;
                ctx.fillRect(-headR*0.8, headY + headR*0.75, headR*1.6, 2.5*s);
                // Gold trim on neck
                ctx.fillStyle = G;
                ctx.fillRect(-headR*0.6, headY + headR*0.7, headR*1.2, 0.8*s);
                // Crest / plume
                ctx.fillStyle = R;
                ctx.beginPath();
                ctx.moveTo(0, headY - headR*1.05);
                ctx.quadraticCurveTo(5*s, headY - headR*1.6, 7*s, headY - headR*0.7);
                ctx.quadraticCurveTo(3*s, headY - headR*0.6, 0, headY - headR*0.4);
                ctx.fill();
                // small secondary feather
                ctx.fillStyle = RD;
                ctx.beginPath();
                ctx.moveTo(1*s, headY - headR*0.95);
                ctx.quadraticCurveTo(3*s, headY - headR*1.2, 4*s, headY - headR*0.6);
                ctx.quadraticCurveTo(1.5*s, headY - headR*0.5, 0, headY - headR*0.5);
                ctx.fill();
            }

            ctx.restore();
        }

        _drawHUD(ctx, s, offX, offY, rows, cs) {
            const hudX = offX + 10;
            const hudY = offY + rows*cs + 6;
            ctx.fillStyle = 'rgba(0,0,0,0.6)';
            ctx.beginPath(); ctx.roundRect(hudX-4, hudY-2, 340, 28, 8); ctx.fill();
            ctx.fillStyle = '#fff';
            ctx.font = '13px "Segoe UI"';
            ctx.textAlign = 'start';
            const pct = s.totalDiamonds > 0 ? Math.round(s.collectedDiamonds/s.totalDiamonds*100) : 0;
            ctx.fillText(`步数: ${s.moves}  |  💎 ${s.collectedDiamonds}/${s.totalDiamonds} (${pct}%)  |  ⭐ ${s.score}分`, hudX, hudY+18);
        }

        _drawVictory(ctx, s) {
            const w = gameCanvas.width, h = gameCanvas.height;
            ctx.fillStyle = 'rgba(0,0,0,0.75)';
            ctx.fillRect(0, 0, w, h);
            // panel
            const pw = 420, ph = 260;
            const px = (w-pw)/2, py = (h-ph)/2;
            ctx.fillStyle = 'rgba(20,25,35,0.95)';
            ctx.beginPath(); ctx.roundRect(px, py, pw, ph, 20); ctx.fill();
            ctx.strokeStyle = '#ffaa00';
            ctx.lineWidth = 2;
            ctx.beginPath(); ctx.roundRect(px, py, pw, ph, 20); ctx.stroke();
            // title
            ctx.fillStyle = '#ffaa00';
            ctx.font = 'bold 32px "Segoe UI"';
            ctx.textAlign = 'center';
            ctx.fillText('🏆 到达出口！', w/2, py+50);
            // stats
            ctx.fillStyle = '#fff';
            ctx.font = '18px "Segoe UI"';
            const pct = Math.round(s._winRate);
            ctx.fillText(`钻石收集: ${s.collectedDiamonds}/${s.totalDiamonds} (${pct}%)`, w/2, py+95);
            ctx.fillText(`步数: ${s.moves}  |  基础分: ${s.score - s._winBonus}`, w/2, py+125);
            ctx.fillStyle = '#ffcc00';
            ctx.fillText(`${s._winMsg}  +${s._winBonus} 分`, w/2, py+160);
            ctx.fillStyle = '#fff';
            ctx.font = 'bold 22px "Segoe UI"';
            ctx.fillText(`总分: ${s.score}`, w/2, py+200);
            // hint
            ctx.fillStyle = '#888';
            ctx.font = '14px "Segoe UI"';
            ctx.fillText(evalMode || demoActive ? '自动进行中...' : '点击"生成新迷宫"开始下一局', w/2, py+235);
            ctx.textAlign = 'start';
        }

        getScore() { return this.state ? this.state.score : 0; }
        getDiamonds() {
            if (!this.state) return { collected: 0, total: 0 };
            return { collected: this.state.collectedDiamonds, total: this.state.totalDiamonds };
        }
        _updateUI() {
            const scoreSpan = document.getElementById('score');
            if (scoreSpan) scoreSpan.innerText = this.state.score;
            const collectedSpan = document.getElementById('collectedDiamonds');
            const totalSpan = document.getElementById('totalDiamonds');
            if (collectedSpan) collectedSpan.innerText = this.state.collectedDiamonds;
            if (totalSpan) totalSpan.innerText = this.state.totalDiamonds;
        }
        recomputeShortestPath() {
            if (!this.state) return [];
            const maze = this.state.maze, start = this.state.player, goal = this.state.goal;
            const h = maze.length, w = maze[0].length;
            const dirs = [[-1,0,'up'],[1,0,'down'],[0,-1,'left'],[0,1,'right']];
            const queue = [{x:start[1], y:start[0], path:[]}];
            const visited = Array(h).fill().map(()=>Array(w).fill(false));
            visited[start[0]][start[1]] = true;
            while (queue.length) {
                let {x,y,path} = queue.shift();
                if (x===goal[1] && y===goal[0]) return path;
                for (let [dy,dx,dir] of dirs) {
                    let ny=y+dy, nx=x+dx;
                    if (nx>=0 && nx<w && ny>=0 && ny<h && maze[ny][nx]===0 && !visited[ny][nx]) {
                        visited[ny][nx]=true;
                        queue.push({x:nx, y:ny, path:[...path,dir]});
                    }
                }
            }
            return [];
        }
    }

    // ==================== 贪吃蛇游戏（脑控优化版） ====================
    class SnakeGame extends BaseGame {
        constructor() {
            super();
            this.state = null;
            this._tickTimer = null;
            this._gameOver = false;
        }

        init() {
            this._stopLoop();
            const gridSize = 25;  // 25×25 on 800×800 canvas → 32px per cell
            const mid = Math.floor(gridSize / 2);
            this.state = {
                snake: [[mid, mid - 2], [mid, mid - 3], [mid, mid - 4]],
                direction: 'right',
                nextDirection: 'right',
                food: [mid, mid + 3],
                score: 0,
                gridSize: gridSize,
                cellSize: Math.floor(gameCanvas.width / gridSize),
                alive: true,
                tickMs: 280,
                growPending: 0
            };
            this._gameOver = false;
            this._spawnFood();
            this._startLoop();
            this._updateUI();
        }

        // ---- game loop ----
        _startLoop() {
            this._stopLoop();
            this._tickTimer = setInterval(() => this._tick(), this.state.tickMs);
        }
        _stopLoop() {
            if (this._tickTimer) { clearInterval(this._tickTimer); this._tickTimer = null; }
        }

        _tick() {
            const s = this.state;
            if (!s || !s.alive) return;

            // Apply queued direction
            s.direction = s.nextDirection;

            // Move head
            const head = s.snake[0];
            let newHead;
            switch (s.direction) {
                case 'up':    newHead = [head[0] - 1, head[1]]; break;
                case 'down':  newHead = [head[0] + 1, head[1]]; break;
                case 'left':  newHead = [head[0], head[1] - 1]; break;
                case 'right': newHead = [head[0], head[1] + 1]; break;
                default: return;
            }

            // Wall collision → game over (no wrap for brain control fairness)
            if (newHead[0] < 0 || newHead[0] >= s.gridSize ||
                newHead[1] < 0 || newHead[1] >= s.gridSize) {
                this._die();
                return;
            }

            // Self-collision → game over
            for (let i = 0; i < s.snake.length; i++) {
                if (s.snake[i][0] === newHead[0] && s.snake[i][1] === newHead[1]) {
                    this._die();
                    return;
                }
            }

            // Move
            s.snake.unshift(newHead);

            // Eat food?
            if (newHead[0] === s.food[0] && newHead[1] === s.food[1]) {
                s.score += 10;
                s.growPending += 1;
                this._spawnFood();
                // Speed up slightly
                if (s.tickMs > 120) {
                    s.tickMs = Math.max(120, s.tickMs - 5);
                    this._startLoop();  // restart timer with new interval
                }
            } else if (s.growPending > 0) {
                s.growPending--;
            } else {
                s.snake.pop();
            }

            this.render(gameCtx);
            this._updateUI();
        }

        _spawnFood() {
            const s = this.state;
            const occupied = new Set(s.snake.map(p => p[0] * 1000 + p[1]));
            const candidates = [];
            for (let r = 0; r < s.gridSize; r++)
                for (let c = 0; c < s.gridSize; c++)
                    if (!occupied.has(r * 1000 + c)) candidates.push([r, c]);
            if (candidates.length > 0)
                s.food = candidates[Math.floor(Math.random() * candidates.length)];
        }

        _die() {
            const s = this.state;
            s.alive = false;
            this._gameOver = true;
            this._stopLoop();
            this.render(gameCtx);
            setTimeout(() => {
                if (this._gameOver && currentGame === 'snake' && !demoActive && !evalMode) {
                    this.init();
                }
            }, 2000);
        }

        // ---- brain / keyboard input ----
        handleMove(cmd) {
            const s = this.state;
            if (!s || !s.alive) return;
            const opposite = { up: 'down', down: 'up', left: 'right', right: 'left' };
            // Prevent 180° reversal
            if (cmd !== opposite[s.direction]) {
                s.nextDirection = cmd;
            }
        }

        // ---- rendering ----
        render(ctx) {
            const s = this.state;
            const cs = s.cellSize;
            const gs = s.gridSize;
            const totalW = gs * cs;
            const offX = (gameCanvas.width - totalW) / 2;
            const offY = (gameCanvas.height - totalW) / 2;

            ctx.clearRect(0, 0, gameCanvas.width, gameCanvas.height);

            // Grid background
            ctx.fillStyle = '#1a1a2e';
            ctx.fillRect(offX, offY, totalW, totalW);

            // Grid lines
            ctx.strokeStyle = '#16213e';
            ctx.lineWidth = 0.5;
            for (let i = 0; i <= gs; i++) {
                ctx.beginPath();
                ctx.moveTo(offX, offY + i * cs);
                ctx.lineTo(offX + totalW, offY + i * cs);
                ctx.stroke();
                ctx.beginPath();
                ctx.moveTo(offX + i * cs, offY);
                ctx.lineTo(offX + i * cs, offY + totalW);
                ctx.stroke();
            }

            // Food (pulsing)
            const pulse = 1 + 0.15 * Math.sin(Date.now() / 200);
            const [fr, fc] = s.food;
            const fx = offX + fc * cs + cs / 2, fy = offY + fr * cs + cs / 2;
            const frr = cs * 0.35 * pulse;
            ctx.beginPath();
            ctx.arc(fx, fy, frr, 0, Math.PI * 2);
            ctx.fillStyle = '#ff4444';
            ctx.fill();
            ctx.shadowColor = '#ff0000';
            ctx.shadowBlur = 12;
            ctx.fill();
            ctx.shadowBlur = 0;

            // Snake body
            for (let i = s.snake.length - 1; i >= 0; i--) {
                const [r, c] = s.snake[i];
                const sx = offX + c * cs + 2, sy = offY + r * cs + 2;
                const sw = cs - 4, sh = cs - 4;
                const t = i / Math.max(1, s.snake.length - 1);  // 0=head, 1=tail

                if (i === 0) {
                    // Head
                    const grad = ctx.createLinearGradient(sx, sy, sx + sw, sy + sh);
                    grad.addColorStop(0, '#00ff88');
                    grad.addColorStop(1, '#00cc66');
                    ctx.fillStyle = grad;
                    ctx.shadowColor = '#00ff88';
                    ctx.shadowBlur = 8;
                } else {
                    // Body segments – gradient from bright green to darker
                    const r = Math.floor(30 + 180 * (1 - t));
                    const g = Math.floor(200 + 55 * t);
                    const b = Math.floor(50 + 50 * t);
                    ctx.fillStyle = `rgb(${r},${g},${b})`;
                    ctx.shadowBlur = 0;
                }
                ctx.beginPath();
                ctx.roundRect(sx, sy, sw, sh, cs * 0.3);
                ctx.fill();
                ctx.shadowBlur = 0;
            }

            // Game over overlay
            if (!s.alive) {
                ctx.fillStyle = 'rgba(0,0,0,0.7)';
                ctx.fillRect(offX, offY, totalW, totalW);
                ctx.fillStyle = '#ff4444';
                ctx.font = 'bold 48px "Segoe UI"';
                ctx.textAlign = 'center';
                ctx.fillText('游戏结束', offX + totalW / 2, offY + totalW / 2 - 10);
                ctx.fillStyle = '#fff';
                ctx.font = '24px "Segoe UI"';
                ctx.fillText(`得分: ${s.score}  |  2秒后重开`, offX + totalW / 2, offY + totalW / 2 + 40);
                ctx.textAlign = 'start';
            }

            // Score on canvas
            ctx.fillStyle = '#fff';
            ctx.font = 'bold 16px "Segoe UI"';
            ctx.textAlign = 'center';
            ctx.fillText(`🐍 贪吃蛇 | 得分: ${s.score}`, gameCanvas.width / 2, offY - 12);
            ctx.textAlign = 'start';
        }

        stop() { this._gameOver = false; this._stopLoop(); }
        getScore() { return this.state ? this.state.score : 0; }
        getDiamonds() { return { collected: 0, total: 0 }; }

        _updateUI() {
            const scoreSpan = document.getElementById('score');
            if (scoreSpan) scoreSpan.innerText = this.state.score;
            const collectedSpan = document.getElementById('collectedDiamonds');
            const totalSpan = document.getElementById('totalDiamonds');
            if (collectedSpan) collectedSpan.innerText = '-';
            if (totalSpan) totalSpan.innerText = '-';
        }
    }

    // ==================== 赛车游戏（脑控优化版） ====================
    // ==================== 赛车游戏（运动想象连续控制版） ====================
    // 控制方式：MI 解码输出 left / right / up / down，连续渐进控制方向与速度
    // left  → 向左微调方向（累积效应）
    // right → 向右微调方向
    // up    → 加速
    // down  → 减速
    class RacingGame extends BaseGame {
        constructor() {
            super();
            this.state = null;
            this._animId = null;
            this._gameOver = false;
            this._lastTime = 0;
        }

        init() {
            this._stopLoop();
            const w = gameCanvas.width;
            const h = gameCanvas.height;
            const roadLeft = w * 0.12;
            const roadRight = w * 0.88;

            this.state = {
                roadLeft: roadLeft,
                roadRight: roadRight,
                roadW: roadRight - roadLeft,   // total drivable width in pixels
                carX: w / 2,                   // car center x (pixel, continuous)
                steerMomentum: 0,              // lateral velocity (px/s), decays over time
                steerAmount: 80,               // px/s added per left/right command (MI incremental)
                steerDecay: 3.0,               // momentum decay rate (per second)
                speed: 180,                    // forward speed (px/s)
                baseSpeed: 180,
                maxSpeed: 500,
                minSpeed: 60,
                accelAmount: 30,               // px/s² per up command
                brakeAmount: 40,               // px/s² per down command
                score: 0,
                distance: 0,
                alive: true,
                obstacles: [],                 // { x, y, w, h }
                obstacleTimer: 0,
                obstacleInterval: 1.2,         // seconds between spawns
                roadOffset: 0,                 // scrolling dashes
                carW: 48,
                carH: 80,
            };
            this._gameOver = false;
            this._lastTime = performance.now();
            this._startLoop();
            this._updateUI();
        }

        _startLoop() {
            this._stopLoop();
            this._lastTime = performance.now();
            const loop = (now) => {
                if (!this.state || !this.state.alive) { this._animId = null; return; }
                this._update(now);
                this.render(gameCtx);
                this._animId = requestAnimationFrame(loop);
            };
            this._animId = requestAnimationFrame(loop);
        }
        _stopLoop() {
            if (this._animId) { cancelAnimationFrame(this._animId); this._animId = null; }
        }

        _update(now) {
            const s = this.state;
            const dt = Math.min((now - this._lastTime) / 1000, 0.1);
            this._lastTime = now;

            // ---- steering with momentum ----
            // Apply momentum (decayed each frame)
            s.steerMomentum *= Math.exp(-s.steerDecay * dt);
            // Clamp tiny momentum to zero (avoid drift)
            if (Math.abs(s.steerMomentum) < 0.5) s.steerMomentum = 0;
            // Update car position
            s.carX += s.steerMomentum * dt;
            // Clamp to road edges (with half-car margin)
            const halfCar = s.carW / 2 + 4;
            s.carX = Math.max(s.roadLeft + halfCar, Math.min(s.roadRight - halfCar, s.carX));
            // Stop momentum if hitting edge
            if (s.carX <= s.roadLeft + halfCar + 1 || s.carX >= s.roadRight - halfCar - 1) {
                s.steerMomentum = 0;
            }

            // ---- forward movement ----
            s.distance += s.speed * dt;
            s.score = Math.floor(s.distance / 10);
            s.roadOffset = (s.roadOffset + s.speed * dt) % 50;

            // ---- obstacles ----
            s.obstacleTimer += dt;
            if (s.obstacleTimer >= s.obstacleInterval) {
                s.obstacleTimer = 0;
                s.obstacleInterval = 0.6 + Math.random() * 1.5 / Math.min(s.speed / s.baseSpeed, 2.5);
                this._spawnObstacle();
            }
            for (let obs of s.obstacles) obs.y += s.speed * dt;
            s.obstacles = s.obstacles.filter(o => o.y < gameCanvas.height + 150);

            // ---- collision (AABB) ----
            const cl = s.carX - s.carW / 2 + 6;
            const cr = s.carX + s.carW / 2 - 6;
            const ct = gameCanvas.height - 130;
            const cb = gameCanvas.height - 30;
            for (let obs of s.obstacles) {
                if (cl < obs.x + obs.w && cr > obs.x && ct < obs.y + obs.h && cb > obs.y) {
                    this._die(); return;
                }
            }
        }

        _spawnObstacle() {
            const s = this.state;
            const margin = s.carW;
            const availWidth = s.roadW - margin * 2;
            // Random x anywhere on road (continuous, not lane-locked)
            const ox = s.roadLeft + margin + Math.random() * availWidth;
            s.obstacles.push({
                x: ox - s.carW / 2,
                y: -120 - Math.random() * 250,
                w: s.carW,
                h: s.carH
            });
        }

        _die() {
            const s = this.state;
            s.alive = false;
            this._gameOver = true;
            this.render(gameCtx);
            setTimeout(() => {
                if (this._gameOver && currentGame === 'racing' && !demoActive && !evalMode) {
                    this.init();
                }
            }, 2000);
        }

        // ---- brain / keyboard (MI-style continuous incremental control) ----
        handleMove(cmd) {
            const s = this.state;
            if (!s || !s.alive) return;
            switch (cmd) {
                case 'left':
                    // Add leftward momentum (cumulative with repeated MI commands)
                    s.steerMomentum -= s.steerAmount;
                    s.steerMomentum = Math.max(s.steerMomentum, -400);
                    break;
                case 'right':
                    s.steerMomentum += s.steerAmount;
                    s.steerMomentum = Math.min(s.steerMomentum, 400);
                    break;
                case 'up':
                    s.speed = Math.min(s.maxSpeed, s.speed + s.accelAmount);
                    break;
                case 'down':
                    s.speed = Math.max(s.minSpeed, s.speed - s.brakeAmount);
                    break;
            }
        }

        // ---- rendering ----
        render(ctx) {
            const s = this.state;
            const w = gameCanvas.width, h = gameCanvas.height;

            ctx.clearRect(0, 0, w, h);

            // Grass
            ctx.fillStyle = '#2d5a27';
            ctx.fillRect(0, 0, w, h);

            // Road
            const roadGrad = ctx.createLinearGradient(s.roadLeft, 0, s.roadRight, 0);
            roadGrad.addColorStop(0, '#3a3a3a');
            roadGrad.addColorStop(0.08, '#555');
            roadGrad.addColorStop(0.92, '#555');
            roadGrad.addColorStop(1, '#3a3a3a');
            ctx.fillStyle = roadGrad;
            ctx.fillRect(s.roadLeft, 0, s.roadRight - s.roadLeft, h);

            // Road edges
            ctx.strokeStyle = '#fff';
            ctx.lineWidth = 4;
            ctx.beginPath(); ctx.moveTo(s.roadLeft, 0); ctx.lineTo(s.roadLeft, h); ctx.stroke();
            ctx.beginPath(); ctx.moveTo(s.roadRight, 0); ctx.lineTo(s.roadRight, h); ctx.stroke();

            // Centre dashes (scrolling)
            ctx.strokeStyle = '#ccc';
            ctx.lineWidth = 2;
            ctx.setLineDash([25, 25]);
            const cx = (s.roadLeft + s.roadRight) / 2;
            ctx.beginPath(); ctx.moveTo(cx, s.roadOffset); ctx.lineTo(cx, h); ctx.stroke();
            ctx.setLineDash([]);

            // Obstacles
            for (let obs of s.obstacles) {
                const hue = (obs.y * 0.3 + Date.now() * 0.01) % 360;
                this._drawCar(ctx, obs.x + obs.w / 2, obs.y + obs.h / 2, obs.w, obs.h, '#e74c3c');
            }

            // Player car
            this._drawCar(ctx, s.carX, h - 80, s.carW, s.carH, '#2196F3');

            // Speed bar (right side)
            const speedPct = (s.speed - s.minSpeed) / (s.maxSpeed - s.minSpeed);
            ctx.fillStyle = 'rgba(0,0,0,0.5)';
            ctx.fillRect(w - 56, 12, 44, h - 24);
            const barH = (h - 24) * speedPct;
            const gradBar = ctx.createLinearGradient(0, h - 12, 0, 12);
            gradBar.addColorStop(0, '#4caf50'); gradBar.addColorStop(0.5, '#ffeb3b'); gradBar.addColorStop(1, '#f44336');
            ctx.fillStyle = gradBar;
            ctx.fillRect(w - 54, h - 12 - barH, 40, barH);
            ctx.fillStyle = '#fff';
            ctx.font = 'bold 12px monospace'; ctx.textAlign = 'center';
            ctx.fillText(Math.floor(s.speed), w - 34, 30);
            ctx.fillText('km/h', w - 34, h - 2);
            ctx.textAlign = 'start';

            // Steering indicator (small bar below speed)
            const steerPct = s.steerMomentum / 400;  // -1..1
            ctx.fillStyle = 'rgba(0,0,0,0.5)';
            ctx.fillRect(w - 70, 48, 72, 10);
            ctx.fillStyle = '#ff9800';
            const indicatorX = w - 34 + steerPct * 30;
            ctx.fillRect(indicatorX - 4, 49, 8, 8);

            // HUD
            ctx.fillStyle = '#fff';
            ctx.font = 'bold 18px "Segoe UI"'; ctx.textAlign = 'center';
            ctx.fillText(`🏎️  得分: ${s.score}  |  距离: ${Math.floor(s.distance)} m`, w / 2, 24);
            ctx.textAlign = 'start';

            // Game over overlay
            if (!s.alive) {
                ctx.fillStyle = 'rgba(0,0,0,0.7)';
                ctx.fillRect(0, 0, w, h);
                ctx.fillStyle = '#ff4444'; ctx.font = 'bold 48px "Segoe UI"'; ctx.textAlign = 'center';
                ctx.fillText('💥 撞车!', w / 2, h / 2 - 20);
                ctx.fillStyle = '#fff'; ctx.font = '24px "Segoe UI"';
                ctx.fillText(`得分: ${s.score}  |  2秒后重开`, w / 2, h / 2 + 30);
                ctx.textAlign = 'start';
            }
        }

        _drawCar(ctx, cx, cy, w, h, color) {
            ctx.save();
            ctx.fillStyle = color;
            ctx.beginPath(); ctx.roundRect(cx - w / 2, cy - h / 2, w, h, 8); ctx.fill();
            ctx.fillStyle = 'rgba(255,255,255,0.25)';
            ctx.fillRect(cx - w * 0.35, cy - h * 0.18, w * 0.7, h * 0.22);
            ctx.fillStyle = '#111';
            ctx.fillRect(cx - w / 2 - 4, cy - h * 0.35, 8, h * 0.22);
            ctx.fillRect(cx - w / 2 - 4, cy + h * 0.13, 8, h * 0.22);
            ctx.fillRect(cx + w / 2 - 4, cy - h * 0.35, 8, h * 0.22);
            ctx.fillRect(cx + w / 2 - 4, cy + h * 0.13, 8, h * 0.22);
            ctx.fillStyle = '#ffff88';
            ctx.fillRect(cx - w * 0.3, cy - h / 2 + 4, w * 0.25, 6);
            ctx.fillRect(cx + w * 0.05, cy - h / 2 + 4, w * 0.25, 6);
            ctx.restore();
        }

        stop() { this._gameOver = false; this._stopLoop(); }
        getScore() { return this.state ? this.state.score : 0; }
        getDiamonds() { return { collected: 0, total: 0 }; }

        _updateUI() {
            const scoreSpan = document.getElementById('score');
            if (scoreSpan) scoreSpan.innerText = this.state.score;
            const collectedSpan = document.getElementById('collectedDiamonds');
            const totalSpan = document.getElementById('totalDiamonds');
            if (collectedSpan) collectedSpan.innerText = '-';
            if (totalSpan) totalSpan.innerText = '-';
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
        if (activeGame) activeGame.stop();  // cleanup previous game's timers
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
        const itr_demo = computeOnlineITR(demoActualSteps);
        const itrStr_demo = itr_demo ? ` | ITR: ${itr_demo.ITR.toFixed(1)} bits/min (${(itr_demo.avgTime*1000).toFixed(0)}ms/trial)` : '';
        if (demoSummaryDiv) {
            demoSummaryDiv.innerHTML = `总尝试: ${totalAttempts} | 正确步数: ${correct} | 准确率: ${accuracy}%${itrStr_demo}`;
        }
        if (demoProgressDiv) {
            demoProgressDiv.innerHTML = `原始路径: ${demoPath.length} 步，实际尝试: ${totalAttempts} 次`;
        }
        if (demoLogDiv) {
            demoLogDiv.innerHTML += `\n🏁 演示结束。总尝试 ${totalAttempts}，正确 ${correct}，准确率 ${accuracy}%${itrStr_demo}`;
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
                        all_confidences: [0,0,0,0],
                        decision_time: 2.0
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

        // 忽略过期消息：结果对应的步骤与当前步骤不一致
        const currentExpected = demoPath[demoCurrentStep];
        if (result.expected !== currentExpected) {
            console.warn(`[Demo] 忽略过期结果: 收到 ${result.expected}，当前期望 ${currentExpected}，步骤 ${demoCurrentStep+1}`);
            return;
        }

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
            decision_time: result.decision_time || 2.0,
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
            const moved = activeGame.handleMove(expected);
            if (!moved) {
                console.error(`❌ 移动 ${expected} 失败，当前坐标 ${activeGame.state.player}，步骤 ${stepNum}，路径索引 ${demoCurrentStep}`);
                alert(`演示出错：步骤 ${stepNum} 无法移动。请检查迷宫数据。`);
                stopDemo();
                return;
            }
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

        if (source === 'offline') {
            demoActive = true;
            demoPath = sequence;
            demoCurrentStep = 0;
            demoActualSteps = [];
            demoStopFlag = false;
            demoRetryCount = 0;
            if (demoLogDiv) demoLogDiv.innerHTML = '演示开始...\n';
            if (demoProgressDiv) demoProgressDiv.innerHTML = `步骤: 0 / ${demoPath.length}`;
            if (demoSummaryDiv) demoSummaryDiv.innerHTML = `标准路径长度: ${demoPath.length} | 实际步数: 0 | 正确率: -`;
            document.getElementById('btn-demo-start').disabled = true;
            document.getElementById('btn-demo-stop').disabled = false;
        }

        const msgType = (source === 'offline') ? "start_offline_sim" : "start_eval";
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
                evalMode = false;
                if (source === 'offline') {
                    demoActive = false;
                    document.getElementById('btn-demo-start').disabled = false;
                    document.getElementById('btn-demo-stop').disabled = true;
                } else {
                    document.getElementById('btn-eval-start').disabled = false;
                    document.getElementById('btn-eval-stop').disabled = true;
                }
                return;
            }
        } else {
            alert('WebSocket 未连接');
            return;
        }

        if (source !== 'offline' && !stimFlashing) startStimuli();

        if (source === 'offline') {
            // 离线演示 UI 已初始化
        } else {
            document.getElementById('eval-info').style.display = 'block';
            document.getElementById('btn-eval-start').disabled = true;
            document.getElementById('btn-eval-stop').disabled = false;
            document.getElementById('eval-info').innerHTML = `评测准备就绪，共 ${sequence.length} 步`;
        }

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

        // 通知后端停止评测
        if (ws && ws.readyState === WebSocket.OPEN) {
            ws.send(JSON.stringify({type: "stop_eval"}));
        }

        if (stimFlashing) stopStimuli();
        if (realtimeActive) stopRealtime();
        if (activeGame) activeGame.render(gameCtx);

        const totalAttempts = evalResults.length;
        let correct = 0;
        for (let s of evalResults) if (s.match) correct++;
        const accuracy = totalAttempts > 0 ? (correct / totalAttempts * 100).toFixed(2) : 0;
        const itr_eval = computeOnlineITR(evalResults);
        const itrStr_eval = itr_eval ? ` | ITR: ${itr_eval.ITR.toFixed(1)} bits/min (${(itr_eval.avgTime*1000).toFixed(0)}ms/trial)` : '';
        const summaryDiv = document.getElementById('demo-summary');
        if (summaryDiv) {
            summaryDiv.innerHTML = `在线评测结果 | 总尝试: ${totalAttempts} | 正确步数: ${correct} | 准确率: ${accuracy}%${itrStr_eval}`;
        }
        console.log(`[评测汇总] 总尝试: ${totalAttempts}, 正确: ${correct}, 准确率: ${accuracy}%${itrStr_eval}`);
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

    function sendEvalTrialStart() {
    if (!evalMode) return;

    if (evalWaitingForResult) {
        console.warn('[前端] 已有等待结果，不重复发送 eval_step');
        return;
    }

    showIndicator = true;
    document.getElementById('eval-info').innerHTML = `📊 步骤 ${evalTrialIndex+1}/${evalSequence.length} | 👀 请注视 ${evalTarget} (尝试 ${evalRetryCount+1})`;

    clearTimeout(evalPromptTimer);
    clearTimeout(evalMainTimer);
    clearTimeout(evalExtraTimer);

    evalPromptTimer = setTimeout(() => {
        if (!evalMode) return;
        showIndicator = false;
        document.getElementById('eval-info').innerHTML = `📊 步骤 ${evalTrialIndex+1}/${evalSequence.length} | 🧠 解码中... (2秒)`;

        if (ws && ws.readyState === WebSocket.OPEN) {
            ws.send(JSON.stringify({ type: "eval_step", direction: evalTarget }));
            console.log(`[前端] 发送 eval_step, expected=${evalTarget}, 重试=${evalRetryCount}`);
        } else {
            console.error('[前端] WebSocket 未连接，无法发送 eval_step');
            // 根据模式处理错误
            if (demoActive) {
                recordDemoStep({ expected: evalTarget, decoded: 'ws_error', match: false, filename: '连接错误', confidence: 0, all_confidences: [0,0,0,0] });
            } else {
                handleEvalResult({ match: false, decoded: 'ws_error', confidence: 0 });
            }
            return;
        }
        evalWaitingForResult = true;
        evalTrialStartTime = Date.now();

        evalMainTimer = setTimeout(() => {
            if (evalWaitingForResult) {
                console.warn('[前端] 等待 eval_result 超时，继续等待后端响应...');
                evalExtraTimer = setTimeout(() => {
                    if (evalWaitingForResult) {
                        console.warn('[前端] 最终超时，强制失败');
                        if (demoActive) {
                            // 离线演示超时，使用 recordDemoStep
                            recordDemoStep({
                                expected: evalTarget,
                                decoded: 'timeout',
                                match: false,
                                filename: '超时',
                                confidence: 0,
                                all_confidences: [0,0,0,0],
                                decision_time: 2.0
                            });
                        } else {
                            handleEvalResult({ match: false, decoded: 'timeout', confidence: 0, decision_time: 2.0 });
                        }
                    }
                }, 1500);
            }
        }, 5000);
    }, 1500);
}

    function handleEvalResult(data) {
        if (!evalMode) {
            console.log('[前端] 评测模式未激活，忽略 eval_result');
            return;
        }

        if (!evalWaitingForResult) {
            console.warn('[前端] 收到 eval_result 但未在等待状态（可能超时后延迟到达），忽略');
            return;
        }

        evalWaitingForResult = false;

        clearTimeout(evalPromptTimer);
        clearTimeout(evalMainTimer);
        if (evalExtraTimer) {
            clearTimeout(evalExtraTimer);
            evalExtraTimer = null;
        }

        if (data.all_confidences) {
            updateConfidenceBars(data.all_confidences);
        }

        const match = data.match;
        const decoded = data.decoded || '无';
        const confidence = (data.confidence * 100).toFixed(1);
        const expected = evalTarget;

        evalResults.push({
            expected: expected,
            decoded: decoded,
            match: match,
            confidence: data.confidence || 0,
            decision_time: data.decision_time || 2.0,
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
    // ==================== 浏览器数据采集 ====================
    let collectMode = false, collectReady = false, collectResolve = null;
    let collectIndex = 0, collectSequence = [];
    const COLLECT_N_PER_DIR = 40;  // 每个方向40个试次，共160个（匹配离线实验）

    async function startCollectMode() {
        if (collectMode) return;
        if (currentMode !== 'online') { alert('请先切换到在线模式'); return; }
        if (!ws || ws.readyState !== WebSocket.OPEN) { alert('WebSocket 未连接'); return; }

        collectMode = true;
        collectReady = false;
        collectIndex = 0;
        document.getElementById('btn-collect-start').disabled = true;
        document.getElementById('btn-collect-stop').disabled = false;
        const infoEl = document.getElementById('collect-info');
        if (infoEl) { infoEl.style.display = 'block'; infoEl.innerHTML = '正在启动采集...'; }

        // 启动采集
        ws.send(JSON.stringify({ type: "start_collect" }));
        try {
            await new Promise((resolve, reject) => {
                collectResolve = resolve;
                setTimeout(() => { if (!collectReady) { collectResolve = null; reject(new Error('start_collect 超时')); } }, 10000);
            });
        } catch (e) { alert('启动采集失败: ' + e.message); stopCollectMode(); return; }

        // 生成采集序列: 每方向N次
        collectSequence = [];
        for (let d = 0; d < 4; d++) {
            for (let i = 0; i < COLLECT_N_PER_DIR; i++) collectSequence.push(d);
        }
        // 随机打乱
        for (let i = collectSequence.length - 1; i > 0; i--) {
            const j = Math.floor(Math.random() * (i + 1));
            [collectSequence[i], collectSequence[j]] = [collectSequence[j], collectSequence[i]];
        }
        collectIndex = 0;
        if (infoEl) infoEl.innerHTML = `采集就绪，共 ${collectSequence.length} 试次`;

        // 开始刺激闪烁
        if (!stimFlashing) startStimuli();
        setTimeout(() => { if (collectMode) runCollectTrial(); }, 500);
    }

    function stopCollectMode() {
        collectMode = false;
        collectReady = false;
        collectResolve = null;
        collectPhase = 'preview';
        document.getElementById('btn-collect-start').disabled = false;
        document.getElementById('btn-collect-stop').disabled = true;
        const infoEl = document.getElementById('collect-info');
        if (infoEl) { infoEl.style.display = 'none'; }
        if (ws && ws.readyState === WebSocket.OPEN) ws.send(JSON.stringify({ type: "stop_collect" }));
        // 恢复正常刺激闪烁
        stimFlashing = false;
        drawStimuli(performance.now());
    }

    // 采集流程状态: 'preview' | 'index' | 'rest' | 'stimulus'
    let collectPhase = 'preview';
    let collectTargetDir = null;

    function runCollectTrial() {
        if (!collectMode || collectIndex >= collectSequence.length) {
            if (collectIndex >= collectSequence.length && collectMode) {
                const infoEl = document.getElementById('collect-info');
                if (infoEl) infoEl.innerHTML = `✅ 采集完成！共 ${collectSequence.length} 试次`;
                setTimeout(() => stopCollectMode(), 2000);
            }
            return;
        }

        const dirIdx = collectSequence[collectIndex];
        const dirs = ['up', 'down', 'left', 'right'];
        collectTargetDir = dirs[dirIdx];
        const infoEl = document.getElementById('collect-info');

        // ---- 阶段1: 预览 (所有块常亮 1s) ----
        collectPhase = 'preview';
        if (infoEl) infoEl.innerHTML = `📥 试次 ${collectIndex + 1}/${collectSequence.length} | 🔍 预览中...`;
        // rAF 循环会检测 collectPhase 并调用 drawAllGray()

        setTimeout(() => {
            if (!collectMode) return;
            // ---- 阶段2: 提示目标方向 (目标块高亮 1s) ----
            collectPhase = 'index';
            if (infoEl) infoEl.innerHTML = `📥 试次 ${collectIndex + 1}/${collectSequence.length} | 👉 目标: ${collectTargetDir}`;
            // rAF 循环会检测 collectPhase 并调用 drawIndexFrame()

            setTimeout(() => {
                if (!collectMode) return;
                // ---- 阶段3: 休息 (十字准星 0.5s) ----
                collectPhase = 'rest';
                if (infoEl) infoEl.innerHTML = `📥 试次 ${collectIndex + 1}/${collectSequence.length} | ✚ 休息`;
                // rAF 循环会检测 collectPhase 并调用 drawRestFrame()

                setTimeout(() => {
                    if (!collectMode) return;
                    // ---- 阶段4: 闪烁采集 (2s) ----
                    collectPhase = 'stimulus';
                    if (infoEl) infoEl.innerHTML = `📥 试次 ${collectIndex + 1}/${collectSequence.length} | ⚡ 采集: ${collectTargetDir}`;
                    stimFlashing = true;
                    stimStartTime = performance.now();
                    lastStimFrameTime = stimStartTime;
                    // rAF 循环检测到 collectPhase==='stimulus' 后恢复正常闪烁绘制
                    // 发送采集指令
                    if (ws && ws.readyState === WebSocket.OPEN) {
                        ws.send(JSON.stringify({ type: "collect_step", direction: collectTargetDir }));
                    }
                }, 500);  // 休息0.5s
            }, 1000);  // 提示1s
        }, 1000);  // 预览1s
    }

    function drawAllGray() {
        stimCtx.clearRect(0, 0, stimCanvas.width, stimCanvas.height);
        for (const dir of dirKeys) {
            const pos = positions[dir];
            if (!pos) continue;
            stimCtx.fillStyle = 'rgb(128, 128, 128)';
            stimCtx.fillRect(pos.x - pos.w/2, pos.y - pos.h/2, pos.w, pos.h);
            stimCtx.fillStyle = '#000';
            const fs = (document.fullscreenElement) ? Math.round(pos.w * 0.35) : 28;
            stimCtx.font = `${fs}px Arial`;
            stimCtx.textAlign = 'center';
            stimCtx.textBaseline = 'middle';
            stimCtx.fillText({up:'↑', down:'↓', left:'←', right:'→'}[dir], pos.x, pos.y);
        }
    }

    function drawIndexFrame(targetDir) {
        stimCtx.clearRect(0, 0, stimCanvas.width, stimCanvas.height);
        for (const dir of dirKeys) {
            const pos = positions[dir];
            if (!pos) continue;
            // 目标块亮白(255)，其余暗灰(60) — 亮度对比已足够指示方向
            const gray = (dir === targetDir) ? 255 : 60;
            stimCtx.fillStyle = `rgb(${gray}, ${gray}, ${gray})`;
            stimCtx.fillRect(pos.x - pos.w/2, pos.y - pos.h/2, pos.w, pos.h);
            stimCtx.fillStyle = (gray > 128) ? '#000' : '#fff';
            const fs = (document.fullscreenElement) ? Math.round(pos.w * 0.35) : 28;
            stimCtx.font = `${fs}px Arial`;
            stimCtx.textAlign = 'center';
            stimCtx.textBaseline = 'middle';
            stimCtx.fillText({up:'↑', down:'↓', left:'←', right:'→'}[dir], pos.x, pos.y);
        }
    }

    function drawRestFrame() {
        stimCtx.clearRect(0, 0, stimCanvas.width, stimCanvas.height);
        const cx = stimCanvas.width / 2, cy = stimCanvas.height / 2;
        const len = Math.min(stimCanvas.width, stimCanvas.height) * 0.08;
        stimCtx.strokeStyle = '#888';
        stimCtx.lineWidth = 3;
        stimCtx.beginPath();
        stimCtx.moveTo(cx - len, cy); stimCtx.lineTo(cx + len, cy);
        stimCtx.moveTo(cx, cy - len); stimCtx.lineTo(cx, cy + len);
        stimCtx.stroke();
    }

    function handleCollectDone(data) {
        if (!collectMode) return;
        collectIndex++;
        // 短暂休息后下一试次
        setTimeout(() => { if (collectMode) runCollectTrial(); }, 500);
    }

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
                } else if (data.type === "eval_result") {
                    handleEvalResult(data);
                } else if (data.type === "realtime_command") {
                    if (lastCmdSpan) lastCmdSpan.innerText = data.command;
                    if (data.all_confidences) updateConfidenceBars(data.all_confidences);
                    if (!evalMode && !demoActive) {
                        handleLocalMove(data.command, true);
                    }
                } else if (data.type === "eval_started") {
                    console.log('[前端] 后端已进入评测模式');
                } else if (data.type === "collect_started") {
                    console.log('[前端] 数据采集已启动');
                    collectReady = true;
                    if (collectResolve) collectResolve();
                } else if (data.type === "collect_done") {
                    console.log('[前端] 采集完成:', data.direction, data.index);
                    handleCollectDone(data);
                } else if (data.type === "collect_error") {
                    console.error('[前端] 采集错误:', data.message);
                } else if (data.type === "collect_stopped") {
                    console.log('[前端] 采集停止, 总计:', data.total);
                } else if (data.type === "trigger_ack") {
                    console.log('[Trigger] 已发送:', data.code);
                } else if (data.type === "offline_status" || data.type === "realtime_status") {
                    // ignore
                } else if (data.command) {
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
        // 侧边栏置信度条（普通模式）
        for (let i = 0; i < dirs.length; i++) {
            const bar = document.getElementById(`bar-${dirs[i]}`);
            const text = document.getElementById(`conf-${dirs[i]}`);
            const percent = Math.round(confidences[i] * 100);
            if (bar) bar.style.width = percent + '%';
            if (text) text.innerText = percent + '%';
        }
    }

    // ==================== 键盘控制 ====================
    function setupKeyboard() {
        document.addEventListener('keydown', (e) => {
            // 演示或评测期间禁止键盘控制
            if (demoActive || evalMode) return;
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
            startEvalMode(null, 'online');
        });

        const evalStopBtn = document.getElementById('btn-eval-stop');
        if (evalStopBtn) evalStopBtn.addEventListener('click', stopEvalMode);

        const collectStartBtn = document.getElementById('btn-collect-start');
        if (collectStartBtn) collectStartBtn.addEventListener('click', startCollectMode);
        const collectStopBtn = document.getElementById('btn-collect-stop');
        if (collectStopBtn) collectStopBtn.addEventListener('click', stopCollectMode);

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

        // 暴露 activeGame 到全局，方便调试
        window.activeGame = activeGame;

        // ========== 全屏切换 ==========
        document.getElementById('btn-fullscreen').addEventListener('click', toggleFullscreen);
        const btnFs = document.getElementById('btn-fullscreen-fs');
        if (btnFs) btnFs.addEventListener('click', toggleFullscreen);
        document.addEventListener('fullscreenchange', onFullscreenChange);
        document.addEventListener('webkitfullscreenchange', onFullscreenChange);
        window.addEventListener('resize', onFullscreenResize);
    }

    window.stopDemo = stopDemo;
    window.startDemo = startDemo;
    window.switchGame = switchGame;
    window.startEvalMode = startEvalMode;
    window.stopEvalMode = stopEvalMode;

    init();
})();