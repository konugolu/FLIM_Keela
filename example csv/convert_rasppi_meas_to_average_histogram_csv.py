import numpy as np

filename = 'example_128bin_irf.csv'  # replace with your actual file path

histograms = []

with open(filename, 'r') as file:
    for line in file:
        # Skip the first 37 lines
        for _ in range(37):
            next(file)
        for line in file:
            if line.startswith('#HLONG'):
                parts = line.strip().split(';')
                # skip the first entry which is the #HLONGXX tag
                bin_values = list(map(int, parts[1:129]))  # take the first 128 bins
                histograms.append(bin_values)
        break  # Exit after processing the relevant lines

# Convert to numpy array for easy averaging
histograms = np.array(histograms)
average_histogram = np.mean(histograms, axis=0) 

# # plot the average histogram for debug
# import matplotlib.pyplot as plt

# plt.figure(figsize=(10, 5))
# plt.plot(average_histogram, marker='o')
# plt.title('Average Histogram')
# plt.xlabel('Bin')
# plt.ylabel('Counts')
# plt.grid(True)
# plt.show()

# Convert to CSV string with integers
average_csv_string = ','.join(str(int(round(val))) for val in average_histogram)

print("output: ", average_csv_string)
with open('test_irf.csv', 'w') as out_file:
    out_file.write(average_csv_string + '\n')
