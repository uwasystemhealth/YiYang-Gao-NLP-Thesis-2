import random

import Utility

total_number_of_records = 696211
number_of_records_to_sample = 360

def generate_random_number():
    random.sample(range(number_of_records), number_of_records_to_sample)
    



if __name__ == "__main__":