from vision import Vision
from optimizer import Tile, Game

class Rummikub():

  def __init__(self):
    self.game = Game()
    self.vision = Vision()

  def rack_from_image(self, img_path, player):

    self.vision.predict(img_path=img_path)

    rack = player.rack
    rack.tiles.clear()
    for box in self.vision.result.boxes:
      tile = Tile.by_name(box.cls[1].upper())
      rack.add_tile(tile)

    return rack

  def board_from_image(self, img_path, game):

    self.vision.predict(img_path=img_path)
    
    board = game.board
    board.tile_sets.clear()
    for box_set in self.vision.result.box_sets:
      tile_names = [box.cls[1].upper() for box in box_set]
      tile_set = board.TileSet.by_names(tile_names)
      board.add_tile_set(tile_set)

    return board
