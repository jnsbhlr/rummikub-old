from enum import Enum
import random
import numpy as np
from itertools import combinations, islice, groupby
import cvxpy as cp
from copy import deepcopy

class TileType(Enum):
  NUMBER = 1
  JOKER = 9

class TileColor(Enum):
  BLACK = 1
  BLUE = 2
  ORANGE = 3
  RED = 4

class SetType(Enum):
  RUN = 1
  GROUP = 2

class Tile():

  @classmethod
  def by_name(cls, name):
    tile_name = name.upper()
    if tile_name == 'JOKER':
      return JokerTile()
    else:
      color, value = tile_name.split('_', 1)
      return NumberTile(TileColor[color], int(value))

  def __eq__(self, other):
    if isinstance(other, self.__class__):
      return self.code==other.code
    return NotImplemented

  def __hash__(self):
    return hash(self.code)

class NumberTile(Tile):

  def __init__(self, color, value):
    self.type = TileType.NUMBER
    self.color = color
    self.value = value
    self.name = f"{self.color.name}_{self.value}"
    self.code = int(f'{self.type.value}{self.color.value}{str(self.value).zfill(2)}')

  def __str__(self):
    return f"[{self.name}]"

  def __repr__(self):
    return f"[{self.name}]"

class JokerTile(Tile):

  def __init__(self):
    self.type = TileType.JOKER
    self.value = 30
    self.name = self.type.name
    self.code = int(f'{self.type.value}000')

  def __str__(self):
    return f"[{self.name}]"

  def __repr__(self):
    return f"[{self.name}]"

class Game():

    def __init__(self,
      deck_tile_types=[TileType.NUMBER, TileType.JOKER],
      deck_tile_colors=[TileColor.BLUE, TileColor.BLACK, TileColor.ORANGE, TileColor.RED],
      deck_tile_numbers=[*range(1, 14)],
      deck_copies=2,
      player_initial_tiles=13,
      player_min_initial_value=30,
      min_set_length=3):

      self.deck_tile_types = deck_tile_types
      self.deck_tile_colors = deck_tile_colors
      self.deck_tile_numbers = deck_tile_numbers
      self.deck_copies = deck_copies
      self.player_initial_tiles = player_initial_tiles
      self.player_min_initial_value = player_min_initial_value
      self.min_set_length = min_set_length

      self.tiles = []
      for copy in range(self.deck_copies):
        for tile_type in self.deck_tile_types:
          if tile_type.name == 'NUMBER':
            for color in self.deck_tile_colors:
              for number in self.deck_tile_numbers:
                tile = NumberTile(color=color, value=number)
                self.tiles.append(tile)
          elif tile_type.name == 'JOKER':
            tile = JokerTile()
            self.tiles.append(tile)
      self.tiles = sorted(self.tiles, key=lambda x: x.code)

      self.tile_map = {}
      self.tile_map_reversed = {}
      for tile in self.tiles:
        self.tile_map[tile.code] = tile.name
        self.tile_map_reversed[tile.name] = tile.code

      tile_sets = set()
      number_tiles = [x for x in self.tiles if x.type == TileType.NUMBER]
      joker_tiles = [x for x in self.tiles if x.type == TileType.JOKER]

      # Get all possible & desirable runs
      runs = set()
      tiles = sorted(set(number_tiles), key=lambda x: x.code)
      for i in range(0, len(tiles)):
        for set_length in range(self.min_set_length, self.min_set_length*2):
          run = sorted(islice(tiles, i, i+set_length, 1), key=lambda x: x.code)
          if len(run) == set_length:
            if all(tile.color == run[0].color for tile in run):
              runs.add(self.Board.Run(tiles=run))
      tile_sets.update(runs)

      # Get all possible & desirable groups
      groups = set()
      tiles = sorted(set(number_tiles), key=lambda x: x.value)
      grps = groupby(tiles, lambda x: x.value)
      for _, grp in grps:
        tiles = list(grp)
        for set_length in range(self.min_set_length, len(self.deck_tile_colors)+1):
          combs = combinations(tiles, set_length)
          for comb in combs:
            groups.add(self.Board.Group(tiles=list(comb)))
      tile_sets.update(groups)

      # Get all possible & desirable joker variations
      joker_sets = set()
      for tile_set in tile_sets:

        match tile_set.type:
          case SetType.RUN:
            if len(tile_set.tiles) > self.min_set_length:
              replacement_positions = range(1, len(tile_set.tiles)-1)
            else:
              replacement_positions = range(len(tile_set.tiles))
          case SetType.GROUP:
            if len(tile_set.tiles) > self.min_set_length:
              replacement_positions = range(0)
            else:
              replacement_positions = range(len(tile_set.tiles))

        variations = {tile_set}
        for joker in joker_tiles:
          for variation in variations.copy():
            for replacement_position in replacement_positions:
              if not variation.tiles[replacement_position] == joker and variation.tiles.count(joker) < len(joker_tiles):
                variation_tiles = deepcopy(variation.tiles)
                variation_tiles[replacement_position] = joker
                match tile_set.type:
                  case SetType.RUN:
                    variations.add(self.Board.Run(tiles=variation_tiles))
                  case SetType.GROUP:
                    variations.add(self.Board.Group(tiles=variation_tiles))

        joker_sets.update(variations)

      tile_sets.update(joker_sets)

      self.tile_sets = list(tile_sets)

      self.players = []
      self.deck = self.Deck(game=self)
      self.board = self.Board(game=self)

    def __str__(self):
      players = '\n\n'.join([str(player) for player in self.players])
      return f"Rummikub Game\n{self.deck}\n{self.board}\n\n{players}"

    def add_player(self):
      player = self.Player(game=self)
      player.initial_draw()
      self.players.append(player)
      return player

    class Deck():

      def __init__(self, game):
        self.game = game
        self.tiles = self.game.tiles.copy()

      def __str__(self):
        tiles = ', '.join([str(tile) for tile in self.tiles])
        return f"Deck ({len(self.tiles)}): [{tiles}]"

      def add_tile(self, tile):
        self.tiles.append(tile)
        return tile

      def remove_tile(self, tile):
        self.tiles.remove(tile)

      def search_tile(self, name):
        for tile in self.tiles:
          if tile.name == name:
            return tile
        return None

    class Board():

      def __init__(self, game):
          self.game = game
          self.tile_sets = []
          self.tiles = []

      def __str__(self):
          tile_sets = '\n'.join([str(tile_set) for tile_set in self.tile_sets])
          return f"Board ({len(self.tile_sets)} Sets / {len(self.tiles)} Tiles):\n{tile_sets}"

      def add_tile_set(self, tile_set):
          self.tile_sets.append(tile_set)
          self.update_tiles()
          return tile_set

      def update_tiles(self):
          self.tiles = []
          for tile_set in self.tile_sets:
            for tile in tile_set.tiles:
              self.tiles.append(tile)

      def search_tile_set(self, name):
          for tile_set in self.tile_sets:
            if tile_set.name == name:
              return tile_set
          return None

      class TileSet():

        @classmethod
        def by_names(cls, names):
            tiles = [Tile.by_name(name) for name in names]
            number_tiles = [tile for tile in tiles if tile.type == TileType.NUMBER]
            if all(tile.color == number_tiles[0].color for tile in number_tiles):
              return Game.Board.Run(tiles)
            elif all(tile.value == number_tiles[0].value for tile in number_tiles):
              return Game.Board.Group(tiles)
            else:
              return Game.Board.TileSet(tiles)

        def __str__(self):
            tile_string = ', '.join([str(tile) for tile in self.tiles])
            return f"{self.type.name}[{tile_string}]"

        def __eq__(self, other):
          if isinstance(other, self.__class__):
            return self.tiles==other.tiles
          return NotImplemented

        def __hash__(self):
          tile_string = ', '.join([str(tile) for tile in self.tiles])
          return hash(f"{self.type.name}[{tile_string}]")

      class Run(TileSet):

          def __init__(self, tiles=[]):
              self.type = SetType.RUN

              jokers = [(i, tile) for i, tile in enumerate(tiles) if tile.type == TileType.JOKER]
              sorted_tiles = sorted([tile for tile in tiles if tile.type == TileType.NUMBER], key=lambda x: x.value)
              for joker in jokers:
                sorted_tiles.insert(joker[0], joker[1])
              self.tiles = sorted_tiles

      class Group(TileSet):

          def __init__(self, tiles=[]):
              self.type = SetType.GROUP
              self.tiles = sorted(tiles, key=lambda x: x.code)

    class Player():

      def __init__(self, game):
          self.game = game
          self.name = f"Player {len(self.game.players)+1}"
          self.rack = self.Rack(player=self)
          self.optimizer = self.Optimizer(player=self)
          self.score = 0
          self.initial_play = True

      def __str__(self):
          return f"{self.name}\n{self.rack}\nScore: {str(self.score)}"

      def draw_tile(self, count=1, verbose=True):
        tiles = []
        for _ in range(count):
          if len(self.game.deck.tiles) == 0:
            print('WARNING: No more tiles in the deck!')
            return
          tile = random.choice(self.game.deck.tiles)
          tiles.append(tile)
          self.game.deck.remove_tile(tile)
          self.rack.add_tile(tile)
        if verbose:
          print(f"Tiles drawn:\n[{', '.join([str(tile) for tile in tiles])}]")
        return tiles

      def initial_draw(self):
        self.draw_tile(count=self.game.player_initial_tiles, verbose=False)

      def update_score(self):
        self.score = 0
        for tile in self.rack.tiles:
          self.score += tile.value

      class Rack():

        def __init__(self, player):
            self.player = player
            self.tiles = []

        def __str__(self):
            tiles = ', '.join([str(tile) for tile in self.tiles])
            return f"Rack ({len(self.tiles)} Tiles):\n[{tiles}]"

        def add_tile(self, tile):
          self.tiles.append(tile)
          self.tiles = sorted(self.tiles, key=lambda x: x.code)
          self.player.update_score()
          return tile

        def remove_tile(self, tile):
          self.tiles.remove(tile)
          self.player.update_score()

        def search_tile(self, name):
          for tile in self.tiles:
            if tile.name == name:
              return tile
          return None

      class Optimizer():

          def __init__(self, player):

              self.player = player

              self.tiles = sorted(self.player.game.tiles, key=lambda x: x.code)
              self.tiles_unique = sorted(set(self.tiles), key=lambda x: x.code)

              self.tiles_count_array = np.array([self.tiles.count(tile) for tile in self.tiles_unique])
              self.tiles_code_array = np.array([tile.code for tile in self.tiles_unique])
              self.tiles_value_array = np.array([tile.value for tile in self.tiles_unique])

              self.sets = self.player.game.tile_sets
              self.sets_matrix = np.array([np.array([tile_set.tiles.count(tile) for tile_set in self.sets]) for tile in self.tiles_unique])

              self.update()

          def update(self):

              self.board = sorted(self.player.game.board.tiles, key=lambda x: x.code)
              self.board_array = np.array([self.board.count(tile) for tile in self.tiles_unique])

              self.rack = sorted(self.player.rack.tiles, key=lambda x: x.code)
              self.rack_array = np.array([self.rack.count(tile) for tile in self.tiles_unique])

          def solve(self):

            self.update()

            sets_matrix = self.sets_matrix
            if self.player.initial_play:
                board_array = np.zeros(self.board_array.shape)
            else:
                board_array = self.board_array
            rack_array = self.rack_array

            value = self.tiles_value_array

            deck_copies = self.player.game.deck_copies

            set_variable = cp.Variable(len(self.sets), integer=True)
            tile_variable = cp.Variable(len(self.tiles_unique), integer=True)

            obj = cp.Maximize(cp.sum(value @ tile_variable))

            constraints = [
                sets_matrix @ set_variable == board_array + tile_variable,
                tile_variable <= rack_array,
                set_variable >= 0, set_variable <= deck_copies,
                tile_variable >= 0, tile_variable <= deck_copies
            ]

            problem = cp.Problem(obj, constraints)
            problem.solve(solver=cp.GLPK_MI)

            if len(list(problem.solution.primal_vars.keys())) == 0 or problem.value == 0:
              return False, 0, np.zeros(len(self.tiles_unique)), np.zeros(len(self.sets))
            elif self.player.initial_play and problem.value < self.player.game.player_min_initial_value:
              return False, problem.value, problem.solution.primal_vars[list(problem.solution.primal_vars.keys())[0]], \
                problem.solution.primal_vars[list(problem.solution.primal_vars.keys())[1]]
            else:
              return True, problem.value, problem.solution.primal_vars[list(problem.solution.primal_vars.keys())[0]], \
                problem.solution.primal_vars[list(problem.solution.primal_vars.keys())[1]]

          def print_solution(self):
            solved, value, tiles, sets = self.solve()
            if solved:
                tile_list = [self.tiles_unique[i] for i, t in enumerate(tiles) for _ in range(int(t))]
                set_list = [self.sets[i] for i, s in enumerate(sets) for _ in range(int(s))]
                print(f"Use the following tiles from your rack:\n{', '.join([str(tile) for tile in tile_list])}")
                print("To form the following sets on the board:")
                for tile_set in set_list:
                    print(tile_set)
            elif self.player.initial_play and value < self.player.game.player_min_initial_value:
                print(f"No solution found for initial move: value {value}")
            else:
                print("No solution found.")
