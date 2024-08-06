
import math
from typing import TypeVar, Generator, List, Tuple, Optional
from collections import deque
import json
from queue import SimpleQueue
import heapq

T = TypeVar("T")  # represents generic type
# represents a Node object (forward-declare to use in Node __init__)
Node = TypeVar("Node")
# represents a custom type used in application
AVLWrappedDictionary = TypeVar("AVLWrappedDictionary")


class Node:
    """
    Implementation of an BST and AVL tree node.
    Do not modify.
    """
    __slots__ = ["value", "parent", "left", "right", "height"]

    def __init__(self, value: T, parent: Node = None,
                 left: Node = None, right: Node = None) -> None:
        """
        Construct an AVL tree node.

        :param value: value held by the node object
        :param parent: ref to parent node of which this node is a child
        :param left: ref to left child node of this node
        :param right: ref to right child node of this node
        """
        self.value = value
        self.parent, self.left, self.right = parent, left, right
        self.height = 0

    def __repr__(self) -> str:
        """
        Represent the AVL tree node as a string.

        :return: string representation of the node.
        """
        return f"<{str(self.value)}>"

    def __str__(self) -> str:
        """
        Represent the AVL tree node as a string.

        :return: string representation of the node.
        """
        return repr(self)


####################################################################################################

class BinarySearchTree:
    """
    Implementation of an BSTree.
    Modify only below indicated line.
    """

    # preallocate storage: see https://stackoverflow.com/questions/472000/usage-of-slots
    __slots__ = ["origin", "size"]

    def __init__(self) -> None:
        """
        Construct an empty BST tree.
        """
        self.origin = None
        self.size = 0

    def __repr__(self) -> str:
        """
        Represent the BSTree as a string.

        :return: string representation of the BST tree
        """
        if self.origin is None:
            return "Empty BST Tree"

        lines = pretty_print_binary_tree(self.origin, 0, False, '-')[0]
        return "\n" + "\n".join((line.rstrip() for line in lines))

    def __str__(self) -> str:
        """
        Represent the BSTree as a string.

        :return: string representation of the BSTree
        """
        return repr(self)

    
    def height(self, root: Node) -> int:
        """
        Calculates the height of a subtree in the AVL tree
        :param root: The root node of the tree.
        :return: the height of the treeINSERT DOCSTRING HERE
        """
        if root is None:
            return -1
        else:
            return root.height

    def insert(self, root: Node, val: T) -> None:
        """
        Inserts a node with the value val into the subtree rooted at root.
        :param root: The root node of the tree
        :param val: The value to be inserted
        """

        if root is None:
            self.origin = Node(val)
            self.origin.height = 0
            self.size = 1
            return
        cur = root
        while True:
            if val < cur.value:
                if cur.left is None:
                    cur.left = Node(val)
                    cur.left.parent = cur
                    break
                else:
                    cur = cur.left
            elif val > cur.value:
                if cur.right is None:
                    cur.right = Node(val)
                    cur.right.parent = cur
                    break
                else:
                    cur = cur.right

        while cur is not None:
            cur.height = 1 + max(self.height(cur.left), self.height(cur.right))
            cur = cur.parent

        self.size += 1

    def remove(self, root: Node, val: T) -> Optional[Node]:
        """
        Removes the node with the value val from the subtree rooted at root.
        :param root: The root node of the tree.
        :param val: The value to be removed from the tree.
        :return: The root node after deletion.
        """
        if not root:
            return None

        if val < root.value:
            root.left = self.remove(root.left, val)
            if root.left:
                root.left.parent = root
        elif val > root.value:
            root.right = self.remove(root.right, val)
            if root.right:
                root.right.parent = root
        else:
            if not root.left or not root.right:
                temp = root.left if root.left else root.right
                if not temp:
                    if root == self.origin:
                        self.origin = None
                    root = None
                else:
                    root = temp
                self.size -= 1
            else:
                temp = root.left
                while temp.right:
                    temp = temp.right
                root.value = temp.value
                root.left = self.remove(root.left, root.value)

        if root and root == self.origin:
            self.origin = root

        if root:
            root.height = 1 + max(self.height(root.left), self.height(root.right))

        return root

    def search(self, root: Node, val: T) -> Optional[Node]:
        """
        searches for the Node with the value val within the subtree rooted at root
        :param root: The root node of the tree.
        :param val: The value to be searched in the tree.
        :return: The node with the specified value, or the node under which val would be inserted as a child.
        """
        curr = root
        last = None
        while curr:
            last = curr
            if curr.value == val:
                return curr
            elif val < curr.value:
                curr = curr.left
            else:
                curr = curr.right
        return last


class AVLTree:
    """
    Implementation of an AVL tree.
    Modify only below indicated line.
    """

    __slots__ = ["origin", "size"]

    def __init__(self) -> None:
        """
        Construct an empty AVL tree.
        """
        self.origin = None
        self.size = 0

    def __repr__(self) -> str:
        """
        Represent the AVL tree as a string.

        :return: string representation of the AVL tree
        """
        if self.origin is None:
            return "Empty AVL Tree"

        return super(AVLTree, self).__repr__()

    def __str__(self) -> str:
        """
        Represent the AVLTree as a string.

        :return: string representation of the BSTree
        """
        return repr(self)


    def height(self, root: Node) -> int:
        """
        Calculates the height of a subtree in the AVL tree
        :param root: The root node of the tree.
        :return: the height of the tree
        """
        if root is None:
            return -1
        return root.height

    def left_rotate(self, root: Node) -> Optional[Node]:
        """
        performs a left rotation on the subtree rooted at root
        :param root: the root node of the tree
        :return: root of the new subtree post-rotation.
        """
        if not root or not root.right:
            return root

        right_child = root.right
        root.right = right_child.left
        if right_child.left:
            right_child.left.parent = root
        right_child.left = root

        if root.parent:
            if root.parent.right == root:
                root.parent.right = right_child
            else:
                root.parent.left = right_child
        else:
            self.origin = right_child

        right_child.parent = root.parent
        root.parent = right_child

        root.height = 1 + max(self.height(root.left), self.height(root.right))
        right_child.height = 1 + max(self.height(right_child.left), self.height(right_child.right))

        return right_child

    def right_rotate(self, root: Node) -> Optional[Node]:
        """
        performs a right rotation on the subtree rooted at root
        :param root: the root node of the tree
        :return: root of the new subtree post-rotation.
        """
        if not root or not root.left:
            return root

        left_child = root.left
        root.left = left_child.right
        if left_child.right:
            left_child.right.parent = root
        left_child.right = root

        if root.parent:
            if root.parent.left == root:
                root.parent.left = left_child
            else:
                root.parent.right = left_child
        else:
            self.origin = left_child

        left_child.parent = root.parent
        root.parent = left_child

        root.height = 1 + max(self.height(root.left), self.height(root.right))
        left_child.height = 1 + max(self.height(left_child.left), self.height(left_child.right))

        return left_child

    def balance_factor(self, root: Node) -> int:
        """
        computes the balance factor of the subtree rooted at root.
        :param root: the root node of the tree
        :return: an integer representing the balance factor of the root
        """
        if root is None:
            return 0
        return self.height(root.left) - self.height(root.right)

    def rebalance(self, root: Node) -> Optional[Node]:
        """
        rebalances the subtree rooted at root if it is unbalanced
        :param root: the root node of the tree
        :return: The root of the new, potentially rebalanced subtree.
        """
        if root is None:
            return None

        bf = self.balance_factor(root)

        if bf > 1 and self.balance_factor(root.left) >= 0:
            return self.right_rotate(root)
        if bf < -1 and self.balance_factor(root.right) <= 0:
            return self.left_rotate(root)
        if bf > 1 and self.balance_factor(root.left) < 0:
            self.left_rotate(root.left)
            return self.right_rotate(root)
        if bf < -1 and self.balance_factor(root.right) > 0:
            self.right_rotate(root.right)
            return self.left_rotate(root)

        return root

    def insert(self, root: Node, val: T) -> Optional[Node]:
        """
        Inserts a node with the value val into the subtree rooted at root.
        :param root: The root node of the tree
        :param val: The value to be inserted
        :return: The root of the new, balanced subtree.
        """
        if root is None:
            new_node = Node(val)
            if self.origin is None:
                self.origin = new_node
            self.size += 1
            return new_node

        if val < root.value:
            root.left = self.insert(root.left, val)
            root.left.parent = root
        elif val > root.value:
            root.right = self.insert(root.right, val)
            root.right.parent = root
        else:
            return root

        root.height = 1 + max(self.height(root.left), self.height(root.right))

        return self.rebalance(root)

    def remove(self, root: Node, val: T) -> Optional[Node]:
        """
        Removes the node with the value val from the subtree rooted at root.
        :param root: The root node of the tree.
        :param val: The value to be removed from the tree.
        :return: The root of the new, balanced subtree.
        """
        if root is None:
            return None

        # Traverse the tree to find the node to be removed
        if val < root.value:
            root.left = self.remove(root.left, val)
            if root.left:
                root.left.parent = root
        elif val > root.value:
            root.right = self.remove(root.right, val)
            if root.right:
                root.right.parent = root
        else:
            # Node with only one child or no child
            if root.left is None or root.right is None:
                temp = root.left if root.left else root.right
                if temp:
                    temp.parent = root.parent  # Update parent pointer

                # Update origin if necessary
                if root == self.origin:
                    self.origin = temp

                self.size -= 1  # Decrement the size of the tree
                return temp  # Return the child to connect with the parent in recursion

            # Node with two children
            else:
                # Find the in-order predecessor (maximum in the left subtree)
                temp = self.max(root.left)
                # Swap values
                root.value = temp.value
                # Remove the in-order predecessor
                root.left = self.remove(root.left, temp.value)
                if root.left:
                    root.left.parent = root

        if root is not None:
            # Update the height of the current node
            root.height = 1 + max(self.height(root.left), self.height(root.right))
            # Rebalance the tree and return the updated node
            root = self.rebalance(root)

        return root



    def min(self, root: Node) -> Optional[Node]:
        """
        Finds the minimum value node in the subtree rooted at the given node.
        :param root: The root node of the subtree.
        :return: The node with the minimum value in the subtree
        """
        cur = root
        if cur:
            while cur.left is not None:
                cur = cur.left
        return cur

    def max(self, root: Node) -> Optional[Node]:
        """
        Finds the maximum value node in the subtree rooted at the given node.
        :param root: The root node of the subtree.
        :return: The node with the maximum value in the subtree
        """
        cur = root
        if cur:
            while cur.right is not None:
                cur = cur.right
        return cur

    def search(self, root: Node, val: T) -> Optional[Node]:
        """
        Searches for a node with a given value in a binary search tree.
        :param root: The root node of the subtree in which to search.
        :param val: The value to search for.
        :return: The node with the given value if it exists in the subtree; otherwise, returns the last node visited.
        """
        if not root or root.value == val:
            return root
        elif root.value > val:
            if root.left is None:
                return root
            return self.search(root.left, val)
        else:
            if root.right is None:
                return root
            return self.search(root.right, val)

    def inorder(self, root: Node) -> Generator[Node, None, None]:
        """
        Performs an inorder traversal of a binary tree.
        :param root: The root node of the subtree to traverse.
        :returns a generator
        """
        if root:
            yield from self.inorder(root.left)
            yield root
            yield from self.inorder(root.right)

    def __iter__(self) -> Generator[Node, None, None]:
        """
        Allows iteration over the tree in an inorder sequence.
        """
        return self.inorder(self.origin)

    def preorder(self, root: Node) -> Generator[Node, None, None]:
        """
        Performs a preorder traversal of a binary tree.
        :param root: The root node of the subtree to traverse.
        :returns a generator
        """
        if root:
            yield root
            yield from self.preorder(root.left)
            yield from self.preorder(root.right)

    def postorder(self, root: Node) -> Generator[Node, None, None]:
        """
        Performs a postorder traversal of a binary tree.
        :param root: The root node of the subtree to traverse.
        :returns a generator
        """
        if root:
            yield from self.postorder(root.left)
            yield from self.postorder(root.right)
            yield root

    def levelorder(self, root: Node) -> Generator[Node, None, None]:
        """
        Performs a level-order traversal of a binary tree.
        :param root: The root node of the subtree to traverse.
        :returns a generator
        """
        if not root:
            return

        queue = SimpleQueue()
        queue.put(root)

        while not queue.empty():
            node = queue.get()
            yield node

            if node.left:
                queue.put(node.left)
            if node.right:
                queue.put(node.right)


####################################################################################################

class User:
    """
    Class representing a user of the stock marker.
    Note: A user can be both a buyer and seller.
    """

    def __init__(self, name, pe_ratio_threshold, div_yield_threshold):
        self.name = name
        self.pe_ratio_threshold = pe_ratio_threshold
        self.div_yield_threshold = div_yield_threshold


####################################################################################################

class Stock:
    __slots__ = ['ticker', 'name', 'price', 'pe', 'mkt_cap', 'div_yield']
    TOLERANCE = 0.001

    def __init__(self, ticker, name, price, pe, mkt_cap, div_yield):
        """
        Initialize a stock.

        :param name: Name of the stock.
        :param price: Selling price of stock.
        :param pe: Price to earnings ratio of the stock.
        :param mkt_cap: Market capacity.
        :param div_yield: Dividend yield for the stock.
        """
        self.ticker = ticker
        self.name = name
        self.price = price
        self.pe = pe
        self.mkt_cap = mkt_cap
        self.div_yield = div_yield

    def __repr__(self):
        """
        Return string representation of the stock.

        :return: String representation of the stock.
        """
        return f"{self.ticker}: PE: {self.pe}"

    def __str__(self):
        """
        Return string representation of the stock.

        :return: String representation of the stock.
        """
        return repr(self)

    def __lt__(self, other):
        """
        Check if the stock is less than the other stock.

        :param other: The other stock to compare to.
        :return: True if the stock is less than the other stock, False otherwise.
        """
        return self.pe < other.pe

    def __eq__(self, other):
        """
        Check if the stock is equal to the other stock.

        :param other: The other stock to compare to.
        :return: True if the stock is equal to the other stock, False otherwise.
        """
        return abs(self.pe - other.pe) < self.TOLERANCE


def make_stock_from_dictionary(stock_dictionary: dict[str: str]) -> Stock:
    """
    Builds an AVL tree with the given stock dictionary.

    :param stock_dictionary: Dictionary of stocks to be inserted into the AVL tree.
    :return: A stock in a Stock object.
    """
    stock = Stock(stock_dictionary['ticker'], stock_dictionary['name'], stock_dictionary['price'], \
                  stock_dictionary['pe_ratio'], stock_dictionary['market_cap'], stock_dictionary['div_yield'])
    return stock


def build_tree_with_stocks(stocks_list: List[dict[str: str]]) -> AVLTree:
    """
    Builds an AVL tree with the given list of stocks.

    :param stocks_list: List of stocks to be inserted into the AVL tree.
    :return: AVL tree with the given stocks.
    """
    avl = AVLTree()
    for stock in stocks_list:
        stock = make_stock_from_dictionary(stock)
        avl.insert(avl.origin, stock)
    return avl


def recommend_stock(stock_tree: AVLTree, user: User, action: str) -> Optional[Stock]:
    """
    Recommends a stock to buy or sell from an AVL tree based on user-defined criteria.
    :param stock_tree: The AVL tree containing stock nodes.
    :param user: The user object containing preferences like P/E ratio and dividend yield thresholds.
    :param action: The action to perform ('buy' or 'sell'), guiding the recommendation criteria.
    :return: the recommended Stock object meeting the user's criteria
    """
    cur = stock_tree.origin
    recommended_stock = None

    while cur:
        if action == 'buy':
            if cur.value.pe <= user.pe_ratio_threshold and cur.value.div_yield >= user.div_yield_threshold:
                recommended_stock = cur.value
                cur = cur.left
            else:
                cur = cur.left if cur.value.pe > user.pe_ratio_threshold else cur.right
        elif action == 'sell':
            if cur.value.pe > user.pe_ratio_threshold or cur.value.div_yield < user.div_yield_threshold:
                recommended_stock = cur.value
                cur = cur.right
            else:
                cur = cur.right if cur.value.pe <= user.pe_ratio_threshold else cur.left

    return recommended_stock


def prune(stock_tree: AVLTree, threshold: float = 0.05) -> None:
    """
    Prunes the AVL tree by removing stocks whose P/E ratio is below a specified threshold.
    :param stock_tree: An instance of an AVLTree representing the stock tree.
    :param threshold: The P/E ratio threshold below which stocks are pruned from the tree.
    :return: None
    """
    stack = []
    node = stock_tree.origin
    nodes_to_prune = []

    for stock in stock_tree:
        if stock.value.pe < threshold:
            nodes_to_prune.append(stock)
    for node in nodes_to_prune:
        stock_tree.remove(stock_tree.origin, node.value)

from collections import Counter


class Blackbox:
    def __init__(self):
        """
        Initialize a minheap.
        """
        self.heap = []

    def store(self, value: T):
        """
        Push a value into the heap while maintaining minheap property.

        :param value: The value to be added.
        """
        heapq.heappush(self.heap, value)

    def get_next(self) -> T:
        """
        Pop minimum from min heap.

        :return: Smallest value in heap.
        """
        return heapq.heappop(self.heap)

    def __len__(self):
        """
        Length of the heap.

        :return: The length of the heap
        """
        return len(self.heap)

    def __repr__(self) -> str:
        """
        The string representation of the heap.

        :return: The string representation of the heap.
        """
        return repr(self.heap)

    __str__ = __repr__


class HuffmanNode:
    __slots__ = ['character', 'frequency', 'left', 'right', 'parent']

    def __init__(self, character, frequency):
        self.character = character
        self.frequency = frequency

        self.left = None
        self.right = None
        self.parent = None

    def __lt__(self, other):
        """
        Checks if node is less than other.

        :param other: The other node to compare to.
        """
        return self.frequency < other.frequency

    def __repr__(self):
        """
        Returns string representation.

        :return: The string representation.
        """
        return '<Char: {}, Freq: {}>'.format(self.character, self.frequency)

    __str__ = __repr__


class HuffmanTree:
    __slots__ = ['root', 'blackbox']

    def __init__(self):
        self.root = None
        self.blackbox = Blackbox()

    def __repr__(self):
        """
        Returns the string representation.

        :return: The string representation.
        """
        if self.root is None:
            return "Empty Tree"

        lines = pretty_print_binary_tree(self.root, 0, False, '-')[0]
        return "\n" + "\n".join((line.rstrip() for line in lines))

    __str__ = __repr__

    def make_char_map(self) -> dict[str: str]:
        """
        Create a binary mapping from the huffman tree.

        :return: Dictionary mapping from characters to "binary" strings.
        """
        mapping = {}

        def traversal(root: HuffmanNode, current_str: str):
            if not root:
                return

            if not root.left and not root.right:
                mapping[root.character] = current_str
                return

            if root.left:
                traversal(root.left, current_str=current_str + '0')

            if root.right:
                traversal(root.right, current_str=current_str + '1')

        traversal(self.root, '')

        return mapping

    def compress(self, input: str) -> tuple[dict[str: str], List[str]]:
        """
        Compress the input data by creating a map via huffman tree.

        :param input: String to compress.
        :return: First value to return is the mapping from characters to binary strings.
        Second value is the compressed data.
        """
        self.build(input)

        mapping = self.make_char_map()

        compressed_data = []

        for char in input:
            compressed_data.append(mapping[char])

        return mapping, compressed_data

    def decompress(self, mapping: dict[str: str], compressed: List[str]) -> str:
        """
        Use the mapping from characters to binary strings to decompress the array of bits.

        :param mapping: Mapping of characters to binary strings.
        :param compressed: Array of binary strings that are encoded.
        """

        reverse_mapping = {v: k for k, v in mapping.items()}

        decompressed = ""

        for encoded in compressed:
            decompressed += reverse_mapping[encoded]

        return decompressed


    def build(self, chars: str) -> None:
        """
        Constructs a Huffman tree based on the input
        :param chars: The input string from which to build the Huffman tree.
        :return: None
        """
        frequency = {}
        for char in chars:
            if char in frequency:
                frequency[char] += 1
            else:
                frequency[char] = 1

        for char, freq in frequency.items():
            node = HuffmanNode(char, freq)
            self.blackbox.store(node)

        while len(self.blackbox) > 1:
            left = self.blackbox.get_next()
            right = self.blackbox.get_next()

            parent = HuffmanNode(None, left.frequency + right.frequency)
            parent.left = left
            parent.right = right

            self.blackbox.store(parent)

        self.root = self.blackbox.get_next()


def pretty_print_binary_tree(root: Node, curr_index: int, include_index: bool = False,
                             delimiter: str = "-", ) -> \
        Tuple[List[str], int, int, int]:
    """
    Taken from: https://github.com/joowani/binarytree

    Recursively walk down the binary tree and build a pretty-print string.
    In each recursive call, a "box" of characters visually representing the
    current (sub)tree is constructed line by line. Each line is padded with
    whitespaces to ensure all lines in the box have the same length. Then the
    box, its width, and start-end positions of its root node value repr string
    (required for drawing branches) are sent up to the parent call. The parent
    call then combines its left and right sub-boxes to build a larger box etc.
    :param root: Root node of the binary tree.
    :type root: binarytree.Node | None
    :param curr_index: Level-order_ index of the current node (root node is 0).
    :type curr_index: int
    :param include_index: If set to True, include the level-order_ node indexes using
        the following format: ``{index}{delimiter}{value}`` (default: False).
    :type include_index: bool
    :param delimiter: Delimiter character between the node index and the node
        value (default: '-').
    :type delimiter:
    :return: Box of characters visually representing the current subtree, width
        of the box, and start-end positions of the repr string of the new root
        node value.
    :rtype: ([str], int, int, int)
    .. _Level-order:
        https://en.wikipedia.org/wiki/Tree_traversal#Breadth-first_search
    """
    if root is None:
        return [], 0, 0, 0

    line1 = []
    line2 = []
    if include_index:
        node_repr = "{}{}{}".format(curr_index, delimiter, root.value)
    else:
        if type(root) == HuffmanNode:
            node_repr = repr(root)
        elif type(root.value) == AVLWrappedDictionary:
            node_repr = f'{root.value},h={root.height},' \
                        f'⬆{str(root.parent.value.key) if root.parent else "None"}'
        else:
            node_repr = f'{root.value},h={root.height},' \
                        f'⬆{str(root.parent.value) if root.parent else "None"}'

    new_root_width = gap_size = len(node_repr)

    # Get the left and right sub-boxes, their widths, and root repr positions
    l_box, l_box_width, l_root_start, l_root_end = pretty_print_binary_tree(
        root.left, 2 * curr_index + 1, include_index, delimiter
    )
    r_box, r_box_width, r_root_start, r_root_end = pretty_print_binary_tree(
        root.right, 2 * curr_index + 2, include_index, delimiter
    )

    # Draw the branch connecting the current root node to the left sub-box
    # Pad the line with whitespaces where necessary
    if l_box_width > 0:
        l_root = (l_root_start + l_root_end) // 2 + 1
        line1.append(" " * (l_root + 1))
        line1.append("_" * (l_box_width - l_root))
        line2.append(" " * l_root + "/")
        line2.append(" " * (l_box_width - l_root))
        new_root_start = l_box_width + 1
        gap_size += 1
    else:
        new_root_start = 0

    # Draw the representation of the current root node
    line1.append(node_repr)
    line2.append(" " * new_root_width)

    # Draw the branch connecting the current root node to the right sub-box
    # Pad the line with whitespaces where necessary
    if r_box_width > 0:
        r_root = (r_root_start + r_root_end) // 2
        line1.append("_" * r_root)
        line1.append(" " * (r_box_width - r_root + 1))
        line2.append(" " * r_root + "\\")
        line2.append(" " * (r_box_width - r_root))
        gap_size += 1
    new_root_end = new_root_start + new_root_width - 1

    # Combine the left and right sub-boxes with the branches drawn above
    gap = " " * gap_size
    new_box = ["".join(line1), "".join(line2)]
    for i in range(max(len(l_box), len(r_box))):
        l_line = l_box[i] if i < len(l_box) else " " * l_box_width
        r_line = r_box[i] if i < len(r_box) else " " * r_box_width
        new_box.append(l_line + gap + r_line)

    # Return the new box, its width and its root repr positions
    return new_box, len(new_box[0]), new_root_start, new_root_end


if __name__ == "__main__":
    pass
