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

async function init() {
  try {
    const response = await fetch("chez.wasm");
    const bytes = await response.arrayBuffer();
    const result = await WebAssembly.instantiate(bytes, {});
    wasm = result.instance.exports;

    wasm.wasm_init_default();
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
    setStatus("Engine thinking...");
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

function engineMove() {
  const depth = parseInt(document.getElementById("depth").value);
  const packed = wasm.wasm_get_best_move(depth);

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
    .map((m) => `${m.num}. ${m.white} ${m.black}`)
    .join("<br>");
  div.innerHTML = `<div class="history-content">${content}</div>`;
  div.scrollTop = div.scrollHeight;
}

function showGameResult(result) {
  document.getElementById("status").classList.remove("thinking", "check");
  document.getElementById("status").classList.add("game-over");

  switch (result) {
    case 1:
      setStatus("White wins by checkmate!");
      break;
    case 2:
      setStatus("Black wins by checkmate!");
      break;
    case 3:
      setStatus("Draw!");
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
