import os
import sys
import cv2
import pytesseract
from distfit import distfit
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from fitter import Fitter, get_common_distributions

all_widths = []
all_heights = []

def process_image(image_path):
    # Load image
    image = cv2.imread(image_path)

    # Convert image to grayscale
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply thresholding to binarize the image
    _, threshold_image = cv2.threshold(gray_image, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

    # Perform OCR using Tesseract with bounding box output
    custom_config = r'--oem 3 --psm 6 -l grci'  # Specify Tesseract OCR options
    data = pytesseract.image_to_data(threshold_image, config=custom_config)

    # Process Tesseract output to extract bounding boxes and recognized text
    for i, line in enumerate(data.split('\n')):
        if i == 0 or not line:
            continue
        values = line.split('\t')
        x, y, w, h = int(values[6]), int(values[7]), int(values[8]), int(values[9])
        conf = int(values[10])
        text = values[-1]

        # Ignore weakly recognized characters
        if conf > 0 and text.strip():
            # Draw bounding box around the character
            cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 1)
            # Print recognized text and width of bounding box (approximate letter width)
            print(f"Letters: {text}, Width: {w}, Height: {h}")
            for _ in range(len(text)):
                all_widths.append(w/len(text))
                all_heights.append(h)
    
    # Display image with bounding boxes
    #cv2.imshow('Image with Bounding Boxes', image)
    #key = cv2.waitKey(0)  # Wait until a key is pressed
    #if key == ord('q'):  # Check if 'q' key is pressed
    #    cv2.destroyAllWindows()  # Close the OpenCV window

# Set Tesseract environment variables
os.environ['TESSDATA_PREFIX'] = r'C:\Program Files (x86)\Tesseract-OCR\tessdata'
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files (x86)\Tesseract-OCR\tesseract'


version = pytesseract.get_tesseract_version()
print("Tesseract version:", version)

def getWeights(input_ruined_image_path, exemplarInputsFilenames, shouldSkipInputImage = False):

    if (shouldSkipInputImage == False):
        process_image(input_ruined_image_path)

    # Process images in the directory
    for i in range(len(exemplarInputsFilenames)):
        if exemplarInputsFilenames[i].lower().endswith(('.png', '.jpg')):
            #image_path = os.path.join(image_dir, filename)
            process_image(exemplarInputsFilenames[i])

    print('\nVASILIS ALL WIDTHS', all_widths)

    # Convert list of widths to a NumPy array
    data = np.array(all_widths)

    # Plot the histogram of the data
    plt.hist(data, bins=30, density=True, alpha=0.6, color='g')

    # Fit data to the best distribution using Fitter
    f = Fitter(data, distributions=get_common_distributions())
    f.fit()
    print("Best fitting distribution:", f.get_best(method='sumsquare_error'))

    # Get the best fitting distribution and its parameters
    best_fit_name, best_fit_params = list(f.get_best(method='sumsquare_error').items())[0]
    dist = getattr(stats, best_fit_name)

    # Extract the parameters
    params = best_fit_params
    print(f"Best fit distribution: {best_fit_name}")
    print(f"Parameters: {params}")

    # Plot the fitted distribution
    x = np.linspace(min(data), max(data), 1000)
    pdf_fitted = dist.pdf(x, **params)
    plt.plot(x, pdf_fitted, 'r-', label=f'Fitted {best_fit_name} Distribution')

    # Define intervals and calculate weights
    mean = params['loc']
    std_dev = params['scale']
    print('VASILIS mean:', mean)
    print('VASILIS std_dev:', std_dev)

    # Define intervals
    intervals = [
        (mean - 2 * std_dev, mean),
        (mean, mean + 2 * std_dev),
        (mean + 2 * std_dev, float('inf')),
        (float('-inf'), mean - 2 * std_dev)
    ]

    # Function to calculate the weight of an interval
    def weight_interval(lower, upper, dist, params):
        if lower == float('-inf'):
            return dist.cdf(upper, **params)
        elif upper == float('inf'):
            return 1 - dist.cdf(lower, **params)
        else:
            return dist.cdf(upper, **params) - dist.cdf(lower, **params)

    # Calculate and print the weights for each interval
    weightsMap = {}
    weights = []
    for lower, upper in intervals:
        weight = weight_interval(lower, upper, dist, params)
        interval_name = f"[{lower:.2f}, {upper:.2f}]" if upper != float('inf') else f"[{lower:.2f}, ∞]"
        weightsMap[interval_name] = weight
        print(f"Weight of interval {interval_name}: {weight:.4f}")
        weights.append(weight)

    # Output weights in a dictionary for better readability
    print("Weights for specified intervals:", weightsMap)

    plt.title('Histogram of Letter Widths with Fitted Best Distribution')
    plt.xlabel('Width')
    plt.ylabel('Density')
    plt.legend()
    plt.show()

    return intervals, weights