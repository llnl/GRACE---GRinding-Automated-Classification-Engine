class PartitionTree:
    class Node:
        def __init__(self, interval, split, depth, df):
            self.interval = interval
            self.split = split
            self.depth = depth
            self.df = df
            self.left = None
            self.right = None

    def __init__(self, df, grit_col, initial_split, min_rows=10):
        self.df = df
        self.grit_col = grit_col
        self.initial_split = initial_split
        self.min_rows = min_rows
        min_val = df[self.grit_col].min()
        max_val = df[self.grit_col].max()
        interval = (min_val, max_val)
        self.root = self.Node(interval, self.initial_split, 0, df)
        self._build_tree(self.root, 0)

    def _build_tree(self, node, depth):
        df = node.df
        interval = node.interval

        min_val = df[self.grit_col].min()
        max_val = df[self.grit_col].max()
        if min_val == max_val or len(df) <= self.min_rows:
            return

        # Use the node's split if set (for root), otherwise find median unique value
        if node.split is not None:
            split = node.split
        else:
            unique_sorted = sorted(df[self.grit_col].unique())
            if len(unique_sorted) < 2:
                return  # Can't split further
            median_idx = len(unique_sorted) // 2
            split = unique_sorted[median_idx]

        # Exclusive split: left is < split, right is >= split
        left_df = df[df[self.grit_col] < split]
        right_df = df[df[self.grit_col] >= split]

        # If either side is empty, try to adjust the split
        if left_df.empty or right_df.empty:
            # Try to find a split that divides the data
            unique_sorted = sorted(df[self.grit_col].unique())
            for candidate in unique_sorted[1:-1]:  # skip min/max
                left_df = df[df[self.grit_col] < candidate]
                right_df = df[df[self.grit_col] >= candidate]
                if not left_df.empty and not right_df.empty:
                    split = candidate
                    break
            else:
                return  # Cannot split further

        node.left = self.Node((interval[0], split), None, depth + 1, left_df)
        node.right = self.Node((split, interval[1]), None, depth + 1, right_df)

        # Recursively split children
        self._build_tree(node.left, depth + 1)
        self._build_tree(node.right, depth + 1)

    def get_all_nodes(self, node=None):
        """
        Returns a list of all nodes in the tree, starting from the given node.
        If node is None, starts from the root.
        """
        if node is None:
            node = self.root
        nodes = [node]
        if node.left is not None:
            nodes.extend(self.get_all_nodes(node.left))
        if node.right is not None:
            nodes.extend(self.get_all_nodes(node.right))
        return nodes

    def get_leaves(self, node=None):
        if node is None:
            node = self.root
        # A node is a leaf if it has no children
        if node.left is None and node.right is None:
            return [node]
        leaves = []
        if node.left is not None:
            leaves.extend(self.get_leaves(node.left))
        if node.right is not None:
            leaves.extend(self.get_leaves(node.right))
        return leaves

    def print_tree(self, node=None, indent=0):
        if node is None:
            node = self.root
        prefix = "    " * indent
        interval_str = f"[{node.interval[0]}, {node.interval[1]})"  # left-inclusive, right-exclusive
        if node.left or node.right:
            print(f"{prefix}Depth {node.depth}: Interval {interval_str}, Split at {node.split}, Rows: {len(node.df)}")
            if node.left:
                self.print_tree(node.left, indent + 1)
            if node.right:
                self.print_tree(node.right, indent + 1)
        else:
            print(f"{prefix}Depth {node.depth}: Interval {interval_str} [Leaf], Rows: {len(node.df)}")

    def find_node_by_interval(self, interval, node=None):
        """
        Returns the node whose interval exactly matches the specified interval.
        If not found, returns None.
        """
        if node is None:
            node = self.root
        if node.interval == interval:
            return node
        # Search left and right children if they exist
        found = None
        if node.left is not None:
            found = self.find_node_by_interval(interval, node.left)
        if found is not None:
            return found
        if node.right is not None:
            found = self.find_node_by_interval(interval, node.right)
        return found