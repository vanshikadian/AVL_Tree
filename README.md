# AVL Tree and Stock Market Analysis

This project implements various data structures and algorithms, including AVL Trees, Huffman Trees, and stock market analysis. It provides utilities for managing stocks, performing operations on AVL Trees, and encoding/decoding strings with Huffman coding.

## Features

- **AVL Tree Implementation:** Insert, remove, and search operations with automatic rebalancing.
- **Stock Market Analysis:** Analyze and recommend stocks based on user-defined criteria.
- **Huffman Coding:** Compress and decompress strings using Huffman trees.
- **Pretty Print Trees:** Visualize binary trees with a clean string representation.

## Installation

Clone this repository to your local machine. No additional libraries are required beyond the Python standard library.

```bash
git clone https://github.com/vanshikadian/AVL_Tree.git
```

## Usage
You can import the classes and functions provided in this project to build and manipulate AVL trees, perform stock analysis, and work with Huffman coding.

## Examples

### AVL Tree Operations
```
from your_module import AVLTree, Node

# Create an AVL tree and insert values
avl_tree = AVLTree()
avl_tree.insert(avl_tree.origin, 10)
avl_tree.insert(avl_tree.origin, 20)
avl_tree.insert(avl_tree.origin, 30)

# Remove a value
avl_tree.remove(avl_tree.origin, 20)

# Search for a value
node = avl_tree.search(avl_tree.origin, 30)
print(f"Found node: {node.value}")
```

### Stock Market Analysis
```
from your_module import User, Stock, build_tree_with_stocks, recommend_stock

# Create a list of stocks
stocks = [
    {'ticker': 'AAPL', 'name': 'Apple', 'price': '150', 'pe_ratio': '30', 'market_cap': '2T', 'div_yield': '1.5'},
    {'ticker': 'GOOGL', 'name': 'Alphabet', 'price': '2800', 'pe_ratio': '25', 'market_cap': '1.5T', 'div_yield': '0'},
]

# Build an AVL tree with stocks
stock_tree = build_tree_with_stocks(stocks)

# Create a user with preferences
user = User(name='Alice', pe_ratio_threshold=28, div_yield_threshold=1.0)

# Recommend a stock to buy
recommended_stock = recommend_stock(stock_tree, user, 'buy')
print(f"Recommended stock to buy: {recommended_stock}")
```

### Huffman Coding

```
from your_module import HuffmanTree

# Create a Huffman tree and compress a string
huffman_tree = HuffmanTree()
mapping, compressed_data = huffman_tree.compress("hello world")

# Decompress the data
decompressed_data = huffman_tree.decompress(mapping, compressed_data)
print(f"Decompressed data: {decompressed_data}")
```


