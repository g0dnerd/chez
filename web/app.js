// Chez Chess Engine - Web Interface

const FULL_PIECE_NAMES = ["pawn", "knight", "bishop", "rook", "queen", "king"];
const PIECE_NAMES = ["", "N", "B", "R", "Q", "K"];
const FILES = "abcdefgh";

let wasm = null;
let selectedSquare = null;
let legalMoves = [];
let lastMoveFrom = null;
let lastMoveTo = null;
let moveHistory = [];
let playerColor = "white";
let pendingPromotion = null;

// Threaded (shared-memory Lazy SMP) state. Only used when the page is
// cross-origin isolated (SharedArrayBuffer available). Falls back to the
// single-threaded chez.wasm otherwise.
let useThreads = false;
let wasmMemory = null; // the shared WebAssembly.Memory (threaded mode only)
let helperWorkers = []; // instantiated helper instances (threaded mode only)
let maxThreads = 1;
let numThreads = 1;

// The active linear memory object. The single-thread module exports its memory;
// the threaded module imports the shared memory we create here.
function wasmMem() {
  return useThreads ? wasmMemory : wasm.memory;
}

// Host clock import for the engine's time control. Absolute (epoch-based) ms so
// the value is comparable across Web Worker instances, which each have their own
// performance.now() origin.
function nowMs() {
  return performance.timeOrigin + performance.now();
}

async function init() {
  try {
    useThreads = self.crossOriginIsolated === true;

    if (useThreads) {
      await initThreaded();
    } else {
      await initSingle();
    }

    await loadNnue();

    wasm.wasm_init_default();
    setupThreadControl();
    updateUI();
    setStatus("Your move");

    document.getElementById("new-game").addEventListener("click", newGame);
    document
      .getElementById("player-color")
      .addEventListener("change", onColorChange);
    document
      .getElementById("unmake-move")
      .addEventListener("click", unmakeMove);

    setupPromotionModal();
  } catch (e) {
    console.error("Failed to load WASM:", e);
    setStatus("Failed to load engine");
  }
}

// Single-threaded engine (no cross-origin isolation): the classic chez.wasm,
// which defines and exports its own linear memory.
async function initSingle() {
  const response = await fetch("chez.wasm");
  const bytes = await response.arrayBuffer();
  const result = await WebAssembly.instantiate(bytes, {
    env: { chez_now_ms: nowMs },
  });
  wasm = result.instance.exports;
}

// Threaded engine: chez-mt.wasm instantiated over one shared WebAssembly.Memory.
// The main instance does I/O (net load) and runs thread 0; helper workers each
// instantiate the same module over the same shared memory and run a helper
// thread. Workers are created up front (over a small memory) and tolerate the
// later growth the main thread triggers when loading the net and allocating the
// search state — shared memory growth is observable across all instances.
async function initThreaded() {
  const response = await fetch("chez-mt.wasm");
  const bytes = await response.arrayBuffer();
  const module = await WebAssembly.compile(bytes);

  // 16 MiB initial; 2 GiB ceiling (matches build.zig max_memory). Grows as the
  // net (~42 MiB) and per-search tables are allocated.
  wasmMemory = new WebAssembly.Memory({
    initial: 256,
    maximum: 32768,
    shared: true,
  });

  const mainInstance = await WebAssembly.instantiate(module, {
    env: { memory: wasmMemory, chez_now_ms: nowMs },
  });
  wasm = mainInstance.exports;

  maxThreads = Math.min(navigator.hardwareConcurrency || 4, 16);
  numThreads = maxThreads;

  // Spawn the helper pool (threads 1..maxThreads-1) and wait until each has
  // instantiated the module over the shared memory.
  const readyPromises = [];
  for (let i = 1; i < maxThreads; i++) {
    const w = new Worker("worker.js");
    helperWorkers.push(w);
    readyPromises.push(
      new Promise((resolve) => {
        w.addEventListener(
          "message",
          (e) => {
            if (e.data.type === "ready") resolve();
          },
          { once: true },
        );
      }),
    );
    w.postMessage({ cmd: "init", module, memory: wasmMemory });
  }
  await Promise.all(readyPromises);
}

// Fetch the NNUE net and hand it to the wasm engine. Non-fatal: if the net is
// missing or fails to parse, the engine keeps evaluating with the hand-crafted
// eval. wasm_nnue_alloc may grow wasm memory (detaching the old ArrayBuffer), so
// the memory view is created only after the allocation, from the current buffer.
async function loadNnue() {
  try {
    const response = await fetch("chez.nnue");
    if (!response.ok) {
      console.warn("No NNUE net found, using hand-crafted eval");
      return;
    }
    const buf = new Uint8Array(await response.arrayBuffer());
    const ptr = wasm.wasm_nnue_alloc(buf.length);
    if (ptr === 0) {
      console.warn("NNUE alloc failed, using hand-crafted eval");
      return;
    }
    new Uint8Array(wasmMem().buffer, ptr, buf.length).set(buf);
    if (wasm.wasm_nnue_load(ptr, buf.length)) {
      console.log("NNUE net loaded");
    } else {
      console.warn("NNUE load failed, using hand-crafted eval");
    }
  } catch (e) {
    console.warn("NNUE load error, using hand-crafted eval:", e);
  }
}

function setupPromotionModal() {
  const modal = document.getElementById("promotion-modal");
  const buttons = modal.querySelectorAll(".promo-btn");

  buttons.forEach((btn) => {
    btn.addEventListener("click", () => {
      if (pendingPromotion) {
        const piece = parseInt(btn.dataset.piece);
        completePromotion(piece);
      }
    });
  });
}

function unmakeMove() {
  const success = wasm.wasm_unmake_move();

  if (success < 0) {
    console.error("Failed to unmake move:", success);
    document
      .getElementById("unmake-move")
      .classList.replace("active", "inactive");
    document.getElementById("unmake-move").disabled = "true";
  }

  selectedSquare = null;
  legalMoves = [];
  lastMoveFrom = null;
  lastMoveTo = null;

  moveHistory.pop();

  if (playerColor == "black") {
    moveHistory[moveHistory.length - 1].black = "";
  }

  updateHistoryDisplay();
  updateUI();

  document
    .getElementById("unmake-move")
    .classList.replace("active", "inactive");
  document.getElementById("unmake-move").disabled = "true";
}

function newGame() {
  wasm.wasm_init_default();
  selectedSquare = null;
  legalMoves = [];
  lastMoveFrom = null;
  lastMoveTo = null;
  moveHistory = [];
  updateHistoryDisplay();
  playerColor = document.getElementById("player-color").value;

  document
    .getElementById("unmake-move")
    .classList.replace("active", "inactive");
  document.getElementById("unmake-move").disabled = "true";

  updateUI();
  setStatus("Your move");

  // If player is black, engine moves first
  if (playerColor === "black") {
    setTimeout(engineMove, 50);
  }
}

function onColorChange() {
  newGame();
}

function renderBoard() {
  const board = document.getElementById("board");
  board.innerHTML = "";

  // Render from rank 8 (top) to rank 1 (bottom)
  for (let rank = 7; rank >= 0; rank--) {
    for (let file = 0; file < 8; file++) {
      const square = file + rank * 8;
      const div = document.createElement("div");
      div.className = "square";
      div.dataset.square = square;

      // Alternate colors
      const isLight = (rank + file) % 2 === 1;
      div.classList.add(isLight ? "light" : "dark");

      // Highlight last move
      if (square === lastMoveFrom || square === lastMoveTo) {
        div.classList.add("last-move");
      }

      // Highlight selected square
      if (square === selectedSquare) {
        div.classList.add("selected");
      }

      // Show legal move indicators
      const legalMove = legalMoves.find((m) => m.end === square);
      if (legalMove) {
        const targetPiece = wasm.wasm_piece_at(square);
        if (targetPiece !== 255) {
          div.classList.add("legal-capture");
        } else {
          div.classList.add("legal-move");
        }
      }

      // Add piece
      const piece = wasm.wasm_piece_at(square);
      if (piece !== 255) {
        const color = wasm.wasm_color_at(square);
        const colorName = color === 0 ? "white" : "black";
        const assetName = `${FULL_PIECE_NAMES[piece]}_${colorName}.svg`;
        div.style = `background-image: url(${assetName})`;
      }
      div.addEventListener("click", () => onSquareClick(square));
      board.appendChild(div);
    }
  }
}

function onSquareClick(square) {
  const result = wasm.wasm_game_result();
  if (result !== 0) return; // Game is over

  const toMove = wasm.wasm_to_move();
  const isPlayerTurn =
    (toMove === 0 && playerColor === "white") ||
    (toMove === 1 && playerColor === "black");

  if (!isPlayerTurn) return;

  // Check if clicking on a legal move destination
  const legalMove = legalMoves.find((m) => m.end === square);
  if (legalMove) {
    // Check for promotion
    const piece = wasm.wasm_piece_at(selectedSquare);
    const endRank = Math.floor(square / 8);
    if (piece === 0 && (endRank === 0 || endRank === 7)) {
      // Pawn promotion - show modal
      pendingPromotion = { start: selectedSquare, end: square };
      showPromotionModal();
      return;
    }

    makeMove(selectedSquare, square, 0);
    return;
  }

  // Check if clicking on own piece
  const piece = wasm.wasm_piece_at(square);
  const color = wasm.wasm_color_at(square);

  if (piece !== 255 && color === toMove) {
    selectedSquare = square;
    generateLegalMovesFor(square);
    renderBoard();
  } else {
    selectedSquare = null;
    legalMoves = [];
    renderBoard();
  }
}

function generateLegalMovesFor(square) {
  legalMoves = [];
  const count = wasm.wasm_generate_moves();

  for (let i = 0; i < count; i++) {
    const packed = wasm.wasm_get_move(i);
    const start = (packed >> 16) & 0xff;
    const end = (packed >> 8) & 0xff;
    const promo = packed & 0xff;

    if (start === square) {
      legalMoves.push({ start, end, promo });
    }
  }
}

function makeMove(start, end, promo) {
  const piece = wasm.wasm_piece_at(start);
  const color = wasm.wasm_color_at(start);
  const captured = wasm.wasm_piece_at(end);

  const success = wasm.wasm_make_move(start, end, promo, true);
  if (success < 0) {
    console.error("Move failed:", start, end, promo, success);
    return;
  }

  // Record move
  const moveStr = formatMove(piece, start, end, captured !== 255, promo);
  addMoveToHistory(moveStr, color);

  lastMoveFrom = start;
  lastMoveTo = end;
  selectedSquare = null;
  legalMoves = [];

  updateUI();

  // Check game result
  const result = wasm.wasm_game_result();
  if (result !== 0) {
    showGameResult(result);
    return;
  }

  // Engine's turn
  const toMove = wasm.wasm_to_move();
  const isEngineTurn =
    (toMove === 0 && playerColor === "black") ||
    (toMove === 1 && playerColor === "white");

  const moveNum = wasm.wasm_fullmove_clock();
  if (moveNum >= 2) {
    document
      .getElementById("unmake-move")
      .classList.replace("inactive", "active");
    document.getElementById("unmake-move").removeAttribute("disabled");
  }

  if (isEngineTurn) {
    setStatus("Chez is thinking...");
    document.getElementById("status").classList.add("thinking");
    setTimeout(engineMove, 50);
  }
}

function showPromotionModal() {
  document.getElementById("promotion-modal").classList.add("active");
}

function hidePromotionModal() {
  document.getElementById("promotion-modal").classList.remove("active");
}

function completePromotion(piece) {
  hidePromotionModal();
  if (pendingPromotion) {
    makeMove(pendingPromotion.start, pendingPromotion.end, piece);
    pendingPromotion = null;
  }
}

// Wire up the thread-count selector (threaded mode only). In single-thread mode
// the control is hidden and numThreads stays 1.
function setupThreadControl() {
  const row = document.getElementById("threads-row");
  if (!row) return;
  if (!useThreads || maxThreads <= 1) {
    row.style.display = "none";
    return;
  }
  const select = document.getElementById("threads");
  select.innerHTML = "";
  for (let n = 1; n <= maxThreads; n++) {
    const opt = document.createElement("option");
    opt.value = String(n);
    opt.textContent = String(n);
    if (n === numThreads) opt.selected = true;
    select.appendChild(opt);
  }
  select.addEventListener("change", () => {
    numThreads = Math.max(1, Math.min(parseInt(select.value) || 1, maxThreads));
  });
}

// Run a time-limited search and return the packed best move. Both the single-
// and multi-threaded paths go through the SMP API (with 1 thread in single mode),
// so iterative deepening stops correctly on the per-move clock.
async function computeBestMove(timeMs) {
  const threads = useThreads ? numThreads : 1;
  const packed = await searchMT(timeMs, threads);
  if (packed !== null) return packed;
  // SMP setup failed (allocation); emergency fixed-depth search so play can go on.
  return wasm.wasm_get_best_move(8);
}

// Shared-memory Lazy SMP search for timeMs milliseconds. Thread 0 runs on this
// (main) thread; helper workers warm the shared TT. Returns the packed move, or
// null if SMP setup failed (caller falls back to a fixed-depth search).
async function searchMT(timeMs, threads) {
  if (!wasm.wasm_smp_begin(timeMs, threads)) return null;

  const helpers = Math.min(threads - 1, helperWorkers.length);
  const donePromises = [];
  for (let i = 1; i <= helpers; i++) {
    const w = helperWorkers[i - 1];
    const stackTop = wasm.wasm_smp_stack_top(i) >>> 0;
    donePromises.push(
      new Promise((resolve) => {
        w.addEventListener(
          "message",
          (e) => {
            if (e.data.type === "done") resolve();
          },
          { once: true },
        );
      }),
    );
    w.postMessage({ cmd: "run", threadId: i, stackTop });
  }

  // Thread 0 runs synchronously here (blocks the main thread for the search,
  // same as the single-thread build). When it returns, signal the helpers to
  // stop and wait for them to finish before reading/freeing the shared state.
  wasm.wasm_smp_run_thread(0);
  wasm.wasm_smp_stop();
  await Promise.all(donePromises);

  return wasm.wasm_smp_finish() >>> 0;
}

async function engineMove() {
  const timeMs = parseInt(document.getElementById("movetime").value);
  const packed = await computeBestMove(timeMs);

  if (packed === 0) {
    // No legal moves
    const result = wasm.wasm_game_result();
    showGameResult(result);
    return;
  }

  const start = (packed >> 16) & 0xff;
  const end = (packed >> 8) & 0xff;
  const promo = packed & 0xff;

  const piece = wasm.wasm_piece_at(start);
  const color = wasm.wasm_color_at(start);
  const captured = wasm.wasm_piece_at(end);

  wasm.wasm_make_move(start, end, promo, false);

  // Record move
  const moveStr = formatMove(piece, start, end, captured !== 255, promo);
  addMoveToHistory(moveStr, color);

  lastMoveFrom = start;
  lastMoveTo = end;

  updateUI();
  document.getElementById("status").classList.remove("thinking");

  // Check game result
  const result = wasm.wasm_game_result();
  if (result !== 0) {
    showGameResult(result);
  } else if (wasm.wasm_in_check() !== 255) {
    setStatus("Check!");
    document.getElementById("status").classList.add("check");
  } else {
    setStatus("Your move");
  }

  const moveNum = wasm.wasm_fullmove_clock();
  if (moveNum >= 2) {
    document
      .getElementById("unmake-move")
      .classList.replace("inactive", "active");
    document.getElementById("unmake-move").removeAttribute("disabled");
  }
}

function formatMove(piece, start, end, isCapture, promo) {
  const startFile = FILES[start % 8];
  const endFile = FILES[end % 8];
  const endRank = Math.floor(end / 8) + 1;

  let str = "";

  // Castling
  if (piece === 5 && Math.abs((start % 8) - (end % 8)) === 2) {
    return end % 8 > 4 ? "O-O" : "O-O-O";
  }

  // Piece letter (not for pawns)
  if (piece !== 0) {
    str += PIECE_NAMES[piece];
  }

  // For pawns, show starting file on captures
  if (piece === 0 && isCapture) {
    str += startFile;
  }

  // Capture symbol
  if (isCapture) {
    str += "x";
  }

  // Destination
  str += endFile + endRank;

  // Promotion
  if (promo !== 0) {
    str += "=" + PIECE_NAMES[promo];
  }

  return str;
}

function addMoveToHistory(moveStr, color) {
  const moveNum = wasm.wasm_fullmove_clock();

  if (color === 0) {
    moveHistory.push({ num: moveNum, white: moveStr, black: "" });
  } else {
    if (moveHistory.length > 0) {
      moveHistory[moveHistory.length - 1].black = moveStr;
    } else {
      moveHistory.push({ num: moveNum, white: "...", black: moveStr });
    }
  }

  updateHistoryDisplay();
}

function updateHistoryDisplay() {
  const div = document.getElementById("history");
  const content = moveHistory
    .map((m) => `<div class="ply">${m.num}. ${m.white} ${m.black}</div>`)
    .join("");
  div.innerHTML = `<div class="history-content">${content}</div>`;
  div.scrollTop = div.scrollHeight;
}

function showGameResult(result) {
  document.getElementById("status").classList.remove("thinking", "check");
  document.getElementById("status").classList.add("game-over");

  switch (result) {
    case 1:
      setStatus("Checkmate - White wins.");
      break;
    case 2:
      setStatus("Checkmate - Black wins.");
      break;
    case 3:
      setStatus("Draw.");
      break;
  }
}

function setStatus(text) {
  document.getElementById("status").textContent = text;
}

function updateUI() {
  renderBoard();

  const toMove = wasm.wasm_to_move();
  document.getElementById("turn").textContent =
    toMove === 0 ? "White" : "Black";
  document.getElementById("move-number").textContent =
    wasm.wasm_fullmove_clock();

  // Update status for check
  const inCheck = wasm.wasm_in_check();
  const statusEl = document.getElementById("status");

  if (inCheck !== 255 && wasm.wasm_game_result() === 0) {
    statusEl.classList.add("check");
    setStatus("Check!");
  } else {
    statusEl.classList.remove("check");
  }
}

// Start the application
init();
