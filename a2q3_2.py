#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 12 18:26:55 2023

@author: xingmingxu
"""

import numpy as np
import pandas as pd
import functools


def normalize(factor: pd.DataFrame):
    """
    Normalizes factor.
    """
    assert("probability" in factor.columns)

    normalize_factor = sum(factor["probability"])
    factor["probability"] = factor["probability"] / normalize_factor

    return factor


def restrict(factor: pd.DataFrame, variable: str, value: int):
    """
    A function that restricts a variable to some value in a given factor.
    """
    new_factor = factor.drop(factor[factor[variable] != value].index)
    new_factor.drop(columns=[variable], inplace=True)
    return new_factor


def sumout(factor: pd.DataFrame, variable: str):
    """
    A function that sums out a variable given a factor.
    """

    new_factor = factor.drop(columns=variable)
    non_var_cols = [col for col in factor.columns if col != variable and
                    col != "probability"]

    new_factor["probability"] = new_factor.groupby(
        non_var_cols)["probability"].transform('sum')
    new_factor = new_factor.drop_duplicates()
    return new_factor



def multiply(factor1: pd.DataFrame, factor2: pd.DataFrame):
    """
    A function that multiplies two factors.
    """
    
    similar_variables = np.intersect1d(
        factor1.columns, factor2.columns)
    similar_variables = np.delete(similar_variables, [-1])
    merged_df = pd.merge(factor1, factor2, on=list(
        similar_variables), how="outer")
    new_probability = np.multiply(merged_df["probability_x"],
                                  merged_df["probability_y"])
    merged_df["probability"] = new_probability
    merged_df.drop(columns=["probability_x", "probability_y"], inplace=True)
    return merged_df


def inference(factorList: list[pd.DataFrame],
              queryVariables: list[str],
              orderedListOfHiddenVariables: list[str],
              evidenceList: list[(str, bool)]):
    """
    A function that computes Pr(queryVariable | evidenceList) 
    by variable elimination. 
    This function should restrict the factors in factorList 
    according to the evidence in evidenceList. 
    Next, it should sum out the hidden variables from the product of the 
    factors in factorList. The variables should be summed out in the order 
    given in orderedListOfHiddenVariables. 
    Finally, the answer should be normalized. 
    """

    def print_factorList(factorList):
        for i in factorList:
            print(i)

    # inference(factorList, P(C, A, B, D, E, F, G),
    # orderedListOfHiddenVariables[A,B,C] , evidenceList (None))

    # First, we restrict all factors.
    new_factors = factorList.copy()
    for var, val in evidenceList:
        for f in range(len(new_factors)):
            if var in new_factors[f].columns:
                print(new_factors[f])
                print("Evidence spotted.")
                new_factors[f] = restrict(new_factors[f], var, val)
                print("Factors after restricting", var)
                print("**************************")
                

    print("Post-restrict:")
    print_factorList(new_factors)

    # Sum out hidden variables
    
    print("**************************")
    for var in orderedListOfHiddenVariables:
        print(var)
        reduced_list = [f for f in new_factors if var in f.columns]
        new_factors = [f for f in new_factors if var not in f.columns]
        temp = functools.reduce(multiply, reduced_list)
        print("New factor from multiplying everything in reduced_list tgt")
        print(temp)
        result = sumout(temp, var)
        print("New factor from summing out", var)
        print(result)
        new_factors.append(result)
        #print(functools.reduce(multiply, reduced_list))
        print("Factors after summing out", var)
        print_factorList(new_factors)
        print("**************************")

    #print(len(new_factors))
    
    # if there are still factors left, reduce.
    # singletons will be dropped by normalize anyways.
    
    new_factors = [f for f in new_factors if f.size > 1]
    
    new_factor = functools.reduce(multiply, new_factors)
    #new_factor = normalize(new_factor)

    print("Pre-normalize:")
    print(new_factor)
    print("Post-normalize:")
    new_factor2 = normalize(new_factor)
    print(new_factor2)
    print("**************************")
    return new_factor2


def partB(factorList):
    """
    Part B.
    
    Our query is P(FH)
    = ∑ P(FH | NDG, FM, FS) P(NDG | FM, NA) P(FB | FS) 
    P(FS) P(FM) P(NA) over NDG, FB, FS, FM, NA
    """

    queryVariables = ["FH"]
    orderedListOfHiddenVariables = ["FB", "NA", "FM", "FS", "NDG"]
    evidenceList = []

    result = inference(factorList, queryVariables,
                       orderedListOfHiddenVariables, evidenceList)
    return result


def partC(factorList):
    """
    Part C.

    Our query is P(FS | FH = True, FM = True) 
    = ∑ P(FH = True | NDG, FS, FM = True) P(NDG | FM = True, NA) P(FS) 
    P(FM = True) P(FB | FS) P(NA) over NDG, NA, FB
    """

    queryVariables = ["FS"]
    orderedListOfHiddenVariables = ["FB", "NA", "NDG"]
    evidenceList = [("FH", True), ("FM", True)]

    result = inference(factorList, queryVariables,
                       orderedListOfHiddenVariables, evidenceList)
    return result


def partD(factorList):
    """
    Part D.
    
    Our query is P(FS | FM = True, FH = True, FB = True)
    = ∑ P(FH | NDG, FS, FM = True) P(NDG | FM = True, NA) P(FS) P(FM) 
    P(FB = True | FS) P(NA) over NDG, NA
    """
    
    queryVariables = ["FS"]
    orderedListOfHiddenVariables = ["NA", "NDG"]
    evidenceList = [("FH", True), ("FM", True), ("FB", True)]

    result = inference(factorList, queryVariables,
                       orderedListOfHiddenVariables, evidenceList)
    return result


def partE(factorList):
    """
    Part E.
    
    Our query is P(FS | FM = True, FH = True, FB = True, NA = True)
    = ∑ P(FH | NDG, FS, FM = True) P(NDG | FM = True, NA = True) P(FS) P(FM) 
    P(FB = True | FS) P(NA = True) over NDG
    """
    
    queryVariables = ["FS"]
    orderedListOfHiddenVariables = ["NDG"]
    evidenceList = [("FH", True), ("FM", True), ("FB", True), ("NA", True)]

    result = inference(factorList, queryVariables,
                       orderedListOfHiddenVariables, evidenceList)
    return result
    


def main() -> None:
    

    na_names = np.array(["NA", "probability"])
    na_data = np.array([np.array([True, 0.3]),
                        np.array([False, 0.7])])
    p_na = pd.DataFrame(data=na_data, columns=na_names)

    fm_names = np.array(["FM", "probability"])
    fm_data = np.array([np.array([True, 1/28]),
                        np.array([False, 27/28])])
    p_fm = pd.DataFrame(data=fm_data, columns=fm_names)

    fs_names = np.array(["FS", "probability"])
    fs_data = np.array([np.array([True, 0.05]),
                        np.array([False, 0.95])])
    p_fs = pd.DataFrame(data=fs_data, columns=fs_names)

    fb_names = np.array(["FB", "FS", "probability"])
    fb_data = np.array([np.array([True, True, 0.6]),
                        np.array([False, True, 0.4]),
                        np.array([True, False, 0.1]),
                        np.array([False, False, 0.9])])
    p_fb = pd.DataFrame(data=fb_data, columns=fb_names)

    ndg_names = np.array(["NDG", "NA", "FM", "probability"])
    ndg_data = np.array([np.array([True, True, True, 0.8]),
                         np.array([False, True, True, 0.2]),
                         np.array([True, False, True, 0.4]),
                         np.array([False, False, True, 0.6]),
                         np.array([True, True, False, 0.5]),
                         np.array([False, True, False, 0.5]),
                         np.array([True, False, False, 0]),
                         np.array([False, False, False, 1])])
    p_ndg = pd.DataFrame(data=ndg_data, columns=ndg_names)

    fh_names = np.array(["FH", "FM", "NDG", "FS", "probability"])
    fh_data = np.array([np.array([True, True, True, True, 0.99]),
                        np.array([False, True, True, True, 0.01]),
                        np.array([True, False, True, True, 0.75]),
                        np.array([False, False, True, True, 0.25]),
                        np.array([True, True, False, True, 0.9]),
                        np.array([False, True, False, True, 0.1]),
                        np.array([True, False, False, True, 0.5]),
                        np.array([False, False, False, True, 0.5]),
                        np.array([True, True, True, False, 0.65]),
                        np.array([False, True, True, False, 0.35]),
                        np.array([True, False, True, False, 0.2]),
                        np.array([False, False, True, False, 0.8]),
                        np.array([True, True, False, False, 0.4]),
                        np.array([False, True, False, False, 0.6]),
                        np.array([True, False, False, False, 0]),
                        np.array([False, False, False, False, 1])])
    p_fh = pd.DataFrame(data=fh_data, columns=fh_names)

    factorList = [p_fh, p_ndg, p_fb, p_fs, p_fm, p_na]
    
    show_b, show_c, show_d, show_e = False, False, True, True
    
    if show_b:
        print("Part (b):")
        result_b = partB(factorList)
    if show_c: 
        print("Part (c):")
        result_c = partC(factorList)
    if show_d:
        print("Part (d):")
        result_d = partD(factorList)
    if show_e:
        print("Part (e):")
        result_e = partE(factorList)
    
    print("Final results:")
    if show_b:
        print("(b)", result_b)
    if show_c:
        print("(c)", result_c)
    if show_d:
        print("(d)", result_d)
    if show_e:
        print("(e)", result_e)

    

if __name__ == "__main__":
    main()
    
