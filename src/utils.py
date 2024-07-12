from typing import TypeVar
import logging

Keys1 = TypeVar("Keys1")
Keys2 = TypeVar("Keys2")
Values = TypeVar("Values")
def swap_dictionary_keys(key1_to_key2_to_value: dict[Keys1,dict[Keys2, Values]]) -> dict[Keys2, dict[Keys1, Values]]:
    """Change 'Key1 -> Key2 -> Value' to 'Key2 -> Key1 -> Value'. 
    
    In more detail: We first generate the product space (Key1, Key2(Key1)). 
    Afterwards we compute the number of entries in (:, Key2).
    Finally we define Key2 -> Key1(Key2) -> Values."""
    #Generate product space
    tuple_set = set()
    length_dict = dict()
    for key1 in key1_to_key2_to_value.keys():
        for key2 in key1_to_key2_to_value[key1].keys():
            tuple_set.add((key2,key1))
            length_dict[key2] = 0
            
    #compute Key1(Key2), i.e., how often does Key2 appear. 
    for key2, key1 in tuple_set:
        length_dict[key2] += 1
    
    key1_to_value = dict()
    key2_to_key1_to_value = dict()
    counter = 0
    for key2, key1 in sorted(tuple_set):
        key1_to_value[key1] =  key1_to_key2_to_value[key1][key2]
        counter += 1
        if counter == length_dict[key2]:
            key2_to_key1_to_value[key2] = key1_to_value
            key1_to_value = dict()
            counter = 0
            
    return key2_to_key1_to_value

def logstring_to_logger(loglevel: str):
    match loglevel:
        case "debug":
            return logging.DEBUG
        case "info":
            return logging.INFO
        case "warning":
            return logging.WARNING
        case "error":
            return logging.ERROR
        case "critical":
            return logging.CRITICAL
        case other:
            msg = "Log level can't be resolved."
            msg += f"\nProvided:\t {loglevel}"
            msg += f"\nExpected: debug, info, warning, error, critical"
            raise ValueError(msg)