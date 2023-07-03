#include <opencv2/opencv.hpp>

using namespace cv;

int main() {
    // Load the Haar cascade XML file for male uniform detection
    CascadeClassifier male_uniform_cascade;
    male_uniform_cascade.load("haarcascades/cascade-mUniform.xml");

    // Load the Haar cascade XML file for female uniform detection
    CascadeClassifier female_uniform_cascade;
    female_uniform_cascade.load("haarcascades/cascade-fUniform.xml");

    // Open the default camera
    VideoCapture cap(0);

    if (!cap.isOpened()) {
        std::cout << "Failed to open the camera" << std::endl;
        return -1;
    }

    // Set the desired frame size
    int new_width = 420;
    int new_height = 340;

    // Read and process frames from the camera
    while (true) {
        // Read a frame from the camera
        Mat frame;
        cap.read(frame);

        // Resize the frame
        resize(frame, frame, Size(new_width, new_height));

        // Convert the frame to grayscale
        Mat gray;
        cvtColor(frame, gray, COLOR_BGR2GRAY);

        // Perform male uniform detection using the male uniform Haar cascade classifier
        std::vector<Rect> male_uniforms;
        male_uniform_cascade.detectMultiScale(gray, male_uniforms, 1.05, 1, 0, Size(20, 20));

        // Perform female uniform detection using the female uniform Haar cascade classifier
        std::vector<Rect> female_uniforms;
        female_uniform_cascade.detectMultiScale(gray, female_uniforms, 1.1, 1, 0, Size(20, 20));

        // Variables to keep track of the uniform with the highest confidence
        double max_confidence_male_uniform = 0.0;
        std::string male_uniform_label = "Male Uniform";

        double max_confidence_female_uniform = 0.0;
        std::string female_uniform_label = "Female Uniform";

        // Iterate over the detected male uniforms and update the maximum confidence if necessary
        for (const Rect& uniform : male_uniforms) {
            // Calculate the confidence level based on the size of the detected region
            double confidence = (uniform.width * uniform.height) / (double)(new_width * new_height);

            if (confidence > max_confidence_male_uniform && confidence > 0) {
                max_confidence_male_uniform = confidence;

                // Draw bounding box
                rectangle(frame, uniform, Scalar(0, 255, 0), 2);

                // Format the confidence as a percentage
                int confidence_percent = static_cast<int>(confidence * 100);

                // Display the label and confidence near the top-left corner of the bounding box
                std::string label_with_confidence = male_uniform_label + " " + std::to_string(confidence_percent) + "%";
                putText(frame, label_with_confidence, Point(uniform.x, uniform.y - 10), FONT_HERSHEY_SIMPLEX, 0.9, Scalar(0, 255, 0), 2);
            }
        }

        // Iterate over the detected female uniforms and update the maximum confidence if necessary
        for (const Rect& uniform : female_uniforms) {
            // Calculate the confidence level based on the size of the detected region
            double confidence = (uniform.width * uniform.height) / (double)(new_width * new_height);

            if (confidence > max_confidence_female_uniform && confidence > 0) {
                max_confidence_female_uniform = confidence;

                // Draw bounding box
                rectangle(frame, uniform, Scalar(255, 0, 0), 2);

                // Format the confidence as a percentage
                int confidence_percent = static_cast<int>(confidence * 100);

                // Display the label and confidence near the top-left corner of the bounding box
                std::string label_with_confidence = female_uniform_label + " " + std::to_string(confidence_percent) + "%";
                putText(frame, label_with_confidence, Point(uniform.x, uniform.y - 10), FONT_HERSHEY_SIMPLEX, 0.9, Scalar(255, 0, 0), 2);
            }
        }

        // Display the frame
        imshow("USTP Uniform Detection", frame);

        // Check for 'q' key press to exit
        if (waitKey(1) == 'q')
            break;
    }

    // Release the camera
    cap.release();

    return 0;
}
