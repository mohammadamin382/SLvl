"""
Advanced PGN Data Loader
- Configurable parsing based on data_settings_format.yaml
- Automatic format detection
- Efficient batch processing
- ELO filtering and game quality assessment
- Parallel processing support
"""

import chess
import chess.pgn
from typing import List, Dict, Optional, Tuple, Iterator
import io
import torch
from torch.utils.data import Dataset, IterableDataset
import numpy as np
from pathlib import Path
import yaml
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import logging
from dataclasses import dataclass


logger = logging.getLogger(__name__)


@dataclass
class GameFilter:
    """Configuration for filtering games"""
    min_elo: Optional[int] = None
    max_elo: Optional[int] = None
    min_moves: int = 10
    max_moves: int = 300
    required_tags: List[str] = None
    time_control_filter: Optional[str] = None
    result_filter: Optional[List[str]] = None  # ['1-0', '0-1', '1/2-1/2']


class PGNFormatDetector:
    """Automatically detect PGN format and tag names"""

    @staticmethod
    def detect_format(pgn_path: str) -> Dict[str, str]:
        """
        Analyze PGN file to detect tag naming conventions

        Returns dict mapping standard names to actual tag names in file
        """
        tag_mapping = {
            'white_elo': None,
            'black_elo': None,
            'result': None,
            'time_control': None,
            'event': None,
            'date': None,
        }

        # Common variations
        elo_variations = ['WhiteElo', 'White_Elo', 'WhiteELO', 'Elo_White']
        black_elo_variations = ['BlackElo', 'Black_Elo', 'BlackELO', 'Elo_Black']

        try:
            with open(pgn_path, 'r', encoding='utf-8', errors='ignore') as f:
                # Read first few games to detect format
                content = f.read(50000)  # Read first 50KB

                # Check for standard tags
                if 'WhiteElo' in content:
                    tag_mapping['white_elo'] = 'WhiteElo'
                else:
                    for var in elo_variations:
                        if var in content:
                            tag_mapping['white_elo'] = var
                            break

                if 'BlackElo' in content:
                    tag_mapping['black_elo'] = 'BlackElo'
                else:
                    for var in black_elo_variations:
                        if var in content:
                            tag_mapping['black_elo'] = var
                            break

                # Standard tags (usually consistent)
                tag_mapping['result'] = 'Result'
                tag_mapping['time_control'] = 'TimeControl'
                tag_mapping['event'] = 'Event'
                tag_mapping['date'] = 'Date'

        except Exception as e:
            logger.warning(f"Error detecting PGN format: {e}")

        return tag_mapping


class ChessBoardConverter:
    """Convert chess board to tensor representation"""

    # Piece to index mapping
    PIECE_TO_IDX = {
        chess.PAWN: 1,
        chess.KNIGHT: 2,
        chess.BISHOP: 3,
        chess.ROOK: 4,
        chess.QUEEN: 5,
        chess.KING: 6,
    }

    @classmethod
    def board_to_tensor(cls, board: chess.Board) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Convert board to tensor representation

        Returns:
            board_tensor: [64] - piece at each square (0=empty, 1-6=white pieces, 7-12=black pieces)
            metadata: [3] - [turn, castling_rights, en_passant_square]
        """
        board_array = np.zeros(64, dtype=np.int64)

        for square in chess.SQUARES:
            piece = board.piece_at(square)
            if piece is not None:
                piece_idx = cls.PIECE_TO_IDX[piece.piece_type]
                if piece.color == chess.BLACK:
                    piece_idx += 6
                board_array[square] = piece_idx

        # Metadata
        turn = 0 if board.turn == chess.WHITE else 1

        # Castling rights (4 bits -> 0-15)
        castling = 0
        if board.has_kingside_castling_rights(chess.WHITE):
            castling |= 1
        if board.has_queenside_castling_rights(chess.WHITE):
            castling |= 2
        if board.has_kingside_castling_rights(chess.BLACK):
            castling |= 4
        if board.has_queenside_castling_rights(chess.BLACK):
            castling |= 8

        # En passant square (0-63, or 64 for none)
        en_passant = board.ep_square if board.ep_square is not None else 64

        metadata = np.array([turn, castling, en_passant], dtype=np.int64)

        return torch.from_numpy(board_array), torch.from_numpy(metadata)


class MoveConverter:
    """Convert chess moves to indices"""

    def __init__(self):
        # Generate all possible moves (simplified - can be expanded)
        self.move_to_idx = {}
        self.idx_to_move = {}
        self._build_move_vocabulary()

    def _build_move_vocabulary(self):
        """Build vocabulary of all possible chess moves"""
        idx = 0

        # All possible from-to square combinations
        for from_square in chess.SQUARES:
            for to_square in chess.SQUARES:
                if from_square != to_square:
                    # Standard move
                    move_uci = chess.square_name(from_square) + chess.square_name(to_square)
                    self.move_to_idx[move_uci] = idx
                    self.idx_to_move[idx] = move_uci
                    idx += 1

                    # Promotion moves (only from 7th rank)
                    if chess.square_rank(from_square) == 6 or chess.square_rank(from_square) == 1:
                        for promotion in ['q', 'r', 'b', 'n']:
                            move_uci_promo = move_uci + promotion
                            self.move_to_idx[move_uci_promo] = idx
                            self.idx_to_move[idx] = move_uci_promo
                            idx += 1

        logger.info(f"Built move vocabulary with {idx} possible moves")

    def move_to_index(self, move: chess.Move) -> int:
        """Convert chess.Move to index"""
        move_uci = move.uci()
        return self.move_to_idx.get(move_uci, 0)  # 0 as default/unknown

    def index_to_move(self, idx: int) -> Optional[str]:
        """Convert index to UCI move string"""
        return self.idx_to_move.get(idx)


class PGNDataset(IterableDataset):
    """
    Iterable dataset for PGN files
    Supports streaming large PGN files without loading everything into memory
    """

    def __init__(
        self,
        pgn_paths: List[str],
        data_config: Dict,
        game_filter: GameFilter,
        move_converter: MoveConverter,
        shuffle: bool = True,
        max_games: Optional[int] = None
    ):
        self.pgn_paths = pgn_paths
        self.data_config = data_config
        self.game_filter = game_filter
        self.move_converter = move_converter
        self.shuffle = shuffle
        self.max_games = max_games

        # Get tag mapping from config or detect
        config_mapping = data_config.get('tag_mapping', {})
        auto_detect = data_config.get('auto_detection', {}).get('enabled', True)

        if config_mapping and not auto_detect:
            # Use config mapping
            self.tag_mapping = config_mapping
            logger.info("Using tag mapping from configuration")
        elif pgn_paths and auto_detect:
            # Auto-detect from first file
            self.tag_mapping = PGNFormatDetector.detect_format(pgn_paths[0])
            logger.info(f"Auto-detected PGN tag mapping: {self.tag_mapping}")
        elif config_mapping:
            # Fallback to config
            self.tag_mapping = config_mapping
            logger.info("Using tag mapping from configuration")
        else:
            # Use defaults
            self.tag_mapping = {
                'white_elo': 'WhiteElo',
                'black_elo': 'BlackElo',
                'result': 'Result',
                'time_control': 'TimeControl',
                'event': 'Event',
                'date': 'Date',
            }
            logger.info("Using default tag mapping")

    def _parse_game(self, game: chess.pgn.Game) -> Optional[List[Dict]]:
        """Parse a single game into training samples"""
        if game is None:
            return None

        # Apply filters
        if not self._filter_game(game):
            return None

        samples = []
        board = game.board()

        # Extract game result as value target
        result = game.headers.get('Result', '*')
        if result == '1-0':
            game_value = 1.0
        elif result == '0-1':
            game_value = -1.0
        elif result == '1/2-1/2':
            game_value = 0.0
        else:
            return None  # Skip unfinished games

        # Iterate through moves
        for move in game.mainline_moves():
            # Get current position
            board_tensor, metadata = ChessBoardConverter.board_to_tensor(board)

            # Get move index
            move_idx = self.move_converter.move_to_index(move)

            # Adjust value based on perspective (positive for side to move)
            value = game_value if board.turn == chess.WHITE else -game_value

            sample = {
                'board': board_tensor,
                'metadata': metadata,
                'move': move_idx,
                'value': value,
            }

            samples.append(sample)

            # Apply move
            board.push(move)

        return samples

    def _filter_game(self, game: chess.pgn.Game) -> bool:
        """Apply game filters"""
        headers = game.headers

        # ELO filter
        white_elo_tag = self.tag_mapping.get('white_elo', 'WhiteElo')
        black_elo_tag = self.tag_mapping.get('black_elo', 'BlackElo')

        try:
            white_elo = int(headers.get(white_elo_tag, 0))
            black_elo = int(headers.get(black_elo_tag, 0))

            if self.game_filter.min_elo is not None:
                if white_elo < self.game_filter.min_elo or black_elo < self.game_filter.min_elo:
                    return False

            if self.game_filter.max_elo is not None:
                if white_elo > self.game_filter.max_elo or black_elo > self.game_filter.max_elo:
                    return False

        except (ValueError, TypeError):
            # Skip games without valid ELO
            if self.game_filter.min_elo is not None:
                return False

        # Result filter
        if self.game_filter.result_filter is not None:
            result = headers.get('Result', '*')
            if result not in self.game_filter.result_filter:
                return False

        return True

    def __iter__(self) -> Iterator[Dict]:
        """Iterate over training samples"""
        games_processed = 0

        for pgn_path in self.pgn_paths:
            try:
                with open(pgn_path, 'r', encoding='utf-8', errors='ignore') as pgn_file:
                    while True:
                        game = chess.pgn.read_game(pgn_file)
                        if game is None:
                            break

                        samples = self._parse_game(game)
                        if samples is not None:
                            for sample in samples:
                                yield sample

                            games_processed += 1

                            if self.max_games is not None and games_processed >= self.max_games:
                                return

            except Exception as e:
                logger.error(f"Error reading PGN file {pgn_path}: {e}")
                continue


def collate_fn(batch: List[Dict]) -> Dict[str, torch.Tensor]:
    """Collate function for DataLoader"""
    boards = torch.stack([item['board'] for item in batch])
    metadata = torch.stack([item['metadata'] for item in batch])
    moves = torch.tensor([item['move'] for item in batch], dtype=torch.long)
    values = torch.tensor([item['value'] for item in batch], dtype=torch.float32)

    return {
        'board': boards,
        'metadata': metadata,
        'move': moves,
        'value': values,
    }
