#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 13 23:31:18 2023

@author: xingmingxu
"""

import math
import numpy as np
import pandas as pd
from scipy.stats import entropy


def transform_response(entry: str):
    return entry == "healthy"


class Node:

    """
    Node class for ID3 tree.
    - attribute represents the attribute the data is being split on.
    - threshold represents the value.
    - truth represents the evaluation if this is a leaf node.
    """

    def __init__(self, attribute=None, threshold=None, left=None, right=None,
                 truth=None):
        self.attribute = attribute
        self.threshold = threshold
        self.left = left
        self.right = right
        self.truth = truth

    def display(self) -> None:
        """
        Pretty-prints the tree this tree represents.
        """
        print(self.attribute, self.threshold)
        self.display(self.left)
        self.display(self.right)


def print_tree(node: Node, level=0) -> None:
    if node != None:
        print_tree(node.left, level + 1)
        print(' ' * 18 * level + '-> ' +
              str(node.attribute) + ", " + str(node.threshold) + "," +
              str(node.truth))
        print_tree(node.right, level + 1)


def build_tree(df: pd.DataFrame) -> Node or None:
    """
    ID3 (Examples, Target_Attribute, Attributes)
    Create a root node for the tree
    If all examples are positive, Return the single-node tree Root, with label = +.
    If all examples are negative, Return the single-node tree Root, with label = -.
    If number of predicting attributes is empty, then Return the single node tree Root,
    with label = most common value of the target attribute in the examples.
    Otherwise Begin
        A ← The Attribute that best classifies examples.
        Decision Tree attribute for Root = A.
        For each possible value, vi, of A,
            Add a new tree branch below Root, corresponding to the test A = vi.
            Let Examples(vi) be the subset of examples that have the value vi for A
            If Examples(vi) is empty
                Then below this new branch add a leaf node with label = most common target value in the examples
            Else below this new branch add the subtree ID3 (Examples(vi), Target_Attribute, Attributes – {A})
    End
    Return Root
    """

    if len(set(df["response"])) == 1:
        return Node(truth=df["response"].unique()[0])

    attributes = list(df.columns)[:-1]
    # print(attributes)

    max_entropy = -1  # A very high value.
    max_attribute = None
    max_threshold = None
    for c in attributes:
        # print(min_attribute, min_threshold, min_entropy)
        # Note: attributes can be reused.
        thresholds = find_splits(sorted(df[c].unique()))

        # print(thresholds)
        for t in thresholds:
            # print(c, t)
            ig = info_gain(df, c, t)
            if ig > max_entropy:
                max_attribute = c
                max_threshold = t
                max_entropy = ig
        # print(c)
        # print(sorted(df[c].unique()))
        # print(thresholds)

    # print("max", max_attribute, max_threshold)
    # return None
    left_node = build_tree(df[df[max_attribute] <= max_threshold])
    right_node = build_tree(df[df[max_attribute] > max_threshold])

    # We have now found the attribute to split on.
    return Node(attribute=max_attribute,
                threshold=max_threshold,
                left=left_node,
                right=right_node)


def info_gain(df: pd.DataFrame, attribute: str, threshold) -> float:
    """Find the information gain by:
        1. Finding the total entropy split on response
        2. Find the remainder
        3. Subtract"""

    # 1. Find the total entropy
    # print(attribute, threshold)
    p_ex = df[df['response'] == "colic."]
    n_ex = df[df['response'] == "healthy."]
    p, n = len(p_ex), len(n_ex)
    assert(p + n == len(df))
    total_entropy = entropy([p/(n+p), n/(n+p)])  # , base)

    # 2. Find the remainder
    attr_true = df[(df[attribute]) <= threshold]
    attr_false = df[(df[attribute]) > threshold]

    tt = attr_true[attr_true['response'] == "colic."]
    tf = attr_true[attr_true['response'] == 'healthy.']
    pt, nt = len(tt), len(tf)
    true_entropy = entropy([pt/(pt+nt), nt/(pt+nt)])  # , base=2)

    ft = attr_false[attr_false['response'] == "colic."]
    ff = attr_false[attr_false['response'] == 'healthy.']
    pf, nf = len(ft), len(ff)
    false_entropy = entropy([pf/(pf+nf), nf/(pf+nf)])  # , base=2)

    # 3. Subtract
    remainder = (pt + nt) / (p + n) * true_entropy + \
        (pf + nf) / (p+n) * false_entropy

    # print(remainder)
    # print(total_entropy - remainder)
    return total_entropy - remainder


def find_splits(unique_values: list):
    """Find the median points in a list of unique values.
        unique_values should be sorted in ascending order."""
    retvals = []
    for i in range(len(unique_values)-1):
        retvals.append((unique_values[i] + unique_values[i+1]) / 2)
    # print(retvals)
    return retvals


def test_entropy(df) -> None:
    """Tests find_entropy function."""

    info_gain(df, "na", lambda x: x > 140)


def tests(df) -> None:
    """Put all the tests in here"""
    tree = build_tree(df)
    # print(tree.left.attribute)
    print_tree(tree, 0)

    # unique_values = [1, 3, 7, 19]
    # find_splits(unique_values)

    # test_entropy(df)


def data_discovery(df: pd.DataFrame) -> None:
    print(df.to_string())
    print(df.columns)
    print(df.shape)  # (132, 17)
    return


def classify(df):
    """Based on the tree.
                                                         -> None, None,colic.
                                        -> na, 141.5,None
                                                          -> None, None,healthy.
                      -> k, 3.55,None
                                                          -> None, None,healthy.
                                        -> gldh, 24.65,None
                                                          -> None, None,colic.
    -> endotoxin, 13.425,None


                      -> None, None,colic.

    """
    print("classify")
    left = (df[df["endotoxin"] <= 13.425])
    ll = left[left["k"] <= 3.55]
    lr = left[left["k"] > 3.55]
    lll = ll[ll["na"] <= 141.5]
    llr = ll[ll["na"] > 141.5]
    lrl = lr[lr["gldh"] <= 24.65]
    lrr = lr[lr["gldh"] > 24.65]
    print(lll["response"])
    print(llr["response"])
    print(lrl["response"])
    print(lrr["response"])
    print(df[df["endotoxin"] > 13.425]["response"])


def main() -> None:

    # Get data
    train_loc = "/Users/xingmingxu/Documents/3B/CS486/A3/horseTrain.txt"
    test_loc = "/Users/xingmingxu/Documents/3B/CS486/A3/horseTest.txt"

    columns = ["k", "na", "cl", "hco3", "endotoxin", "aniongap", "pla2",
               "sdh", "gldh", "tpp", "breath_rate", "pcv", "pulse_rate",
               "fibrinogen", "dimer", "fibperdim", "response"]
    # print(len(columns))

    train = pd.read_csv(train_loc, header=None)
    train.columns = columns
    test = pd.read_csv(test_loc, header=None)
    test.columns = columns

    # data_discovery(train)
    # data_discovery(test)

    tests(train)
    print("train")
    classify(train)
    print("test")
    classify(test)


if __name__ == "__main__":
    main()
