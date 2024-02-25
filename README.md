# Rummikub

Rummikub is a comprehensive artificial intelligence (AI) system designed to engage in the board game Rummikub by leveraging visual analysis and strategic optimization techniques. The system is comprised of a vision module to interpret the current game state and an optimization algorithm designed to propose next best moves aimed at minimizing the player's residual score given a current game state. 

## Usage

You can use the demo Jupyter [notebook](https://github.com/jnsbhlr/rummikub/blob/main/demo/demo-notebook.ipynb) to try it out!

### Instantiate Classes

```python
rummikub = Rummikub()
vision = rummikub.vision
game = rummikub.game
player = game.add_player()
```

### Predict Gamestate

```python
rack = rummikub.rack_from_image(img_path, player) 
```

```python
board = rummikub.board_from_image(img_path, game)
```

### Diplay Current Gamestate

```python
print(game)
```

### Optimizer Move Prediction

```python
player.solver.print_solution()
```

## License

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)