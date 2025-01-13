import numpy as np
import os

def partition_train_valid_test_dup_mgf(input_file, prob):
  print("partition_train_valid_test_dup_mgf()")

  print("input_file = ", os.path.join(input_file))
  print("prob = ", prob)
  
  #output_file_train = input_file + ".train" + ".dup"
  output_file_train = input_file.replace('.mgf', '') + ".train" + ".mgf"
  #output_file_valid = input_file + ".valid" + ".dup"
  output_file_valid = input_file.replace('.mgf','') + ".valid" + ".mgf"
  #output_file_test = input_file + ".test" + ".dup"
  output_file_test = input_file.replace('.mgf', '') + ".test" + ".mgf"

  with open(input_file, mode="r") as input_handle:
    with open(output_file_train, mode="w") as output_handle_train:
      with open(output_file_valid, mode="w") as output_handle_valid:
        with open(output_file_test, mode="w") as output_handle_test:
          counter = 0
          counter_train = 0
          counter_valid = 0
          counter_test = 0
          line = input_handle.readline()
          while line:
            if "BEGIN IONS" in line: # a spectrum found
              counter += 1
              set_num = np.random.choice(a=3, size=1, p=prob)
              if set_num == 0:
                output_handle = output_handle_train
                counter_train += 1
              elif set_num == 1:
                output_handle = output_handle_valid
                counter_valid += 1
              else:
                output_handle = output_handle_test
                counter_test += 1
            output_handle.write(line)
            line = input_handle.readline()

  input_handle.close()
  output_handle_train.close()
  output_handle_valid.close()
  output_handle_test.close()
  
  print("the number of spectra ", counter)
  print("the number of spectrav train set ", counter_train)
  print("the number of spectra valid set", counter_valid)
  print("the number of spectra test set ", counter_test)
  
  
input_file = input("Please enter the path to the MGF file: ")

train_ratio = float(input("Enter the train split ratio (e.g., 0.7 for 70%): "))
test_ratio = float(input("Enter the test split ratio (e.g., 0.15 for 15%): "))
valid_ratio = float(input("Enter the validation split ratio (e.g., 0.15 for 15%): "))

# Validate that the ratios sum to 1
if train_ratio + test_ratio + valid_ratio != 1.0:
    print("Error: The split ratios must sum to 1.0")
else:

    partition_train_valid_test_dup_mgf(input_file, [train_ratio, valid_ratio, test_ratio])


