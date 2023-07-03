#include <opencv2/opencv.hpp>

using namespace cv;

int main() {
   // Load the Haar cascade XML file for male uniform detection
   CascadeClassifier male_uniform_cascade;
   male_uniform_cascade.load("haarcascades/haarcascade-maleUniform.xml");

   // Load the Haar cascade XML file for female uniform detection
   CascadeClassifier female_uniform_cascade;
   female_uniform_cascade.load("haarcascades/haarcascade-femaleUniform.xml");

   // Load the image
   Mat image = imread("samples/male/2.jpg");

   if (image.empty()) {
       std::cout << "Failed to open or find the image" << std::endl;
       return -1;
   }

   // Set the desired frame size
   int new_width = 620;
   int new_height = 540;

   // Resize the image
   resize(image, image, Size(new_width, new_height));

   // Convert the image to grayscale
   Mat gray;
   cvtColor(image, gray, COLOR_BGR2GRAY);

   // Perform male uniform detection using the male uniform Haar cascade classifier
   std::vector<Rect> male_uniforms;
   male_uniform_cascade.detectMultiScale(gray, male_uniforms, 1.01, 1, 0, Size(20, 20));

   // Perform female uniform detection using the female uniform Haar cascade classifier
   std::vector<Rect> female_uniforms;
   female_uniform_cascade.detectMultiScale(gray, female_uniforms, 1.01, 1, 0, Size(20, 20));

   // Variables to keep track of the uniform with the highest confidence
   double max_confidence_male_uniform = 0.0;
   std::string male_uniform_label = "Male Uniform";
   Rect max_confidence_male_rect;

   double max_confidence_female_uniform = 0.0;
   std::string female_uniform_label = "Female Uniform";
   Rect max_confidence_female_rect;

   // Iterate over the detected male uniforms and update the maximum confidence if necessary
   for (const Rect& uniform : male_uniforms) {
       // Calculate the confidence level based on the size of the detected region
       double confidence = (uniform.width*uniform.height) / (double)(new_width*new_height);

       if (confidence > max_confidence_male_uniform && confidence > 0.1) {
           max_confidence_male_uniform = confidence;
           max_confidence_male_rect = uniform;
       }
   }

   // Iterate over the detected female uniforms and update the maximum confidence if necessary
   for (const Rect& uniform : female_uniforms) {
       // Calculate the confidence level based on the size of the detected region
       double confidence = (uniform.width*uniform.height) / (double)(new_width*new_height);

       if (confidence > max_confidence_female_uniform && confidence > 0) {
           max_confidence_female_uniform = confidence;
           max_confidence_female_rect = uniform;
       }
   }

   // Draw bounding box and label for the uniform with the highest confidence
   if (max_confidence_male_uniform > max_confidence_female_uniform) {
       // Draw bounding box
       rectangle(image, max_confidence_male_rect, Scalar(0, 255, 0), 2);

       // Format the confidence as a percentage
       int confidence_percent = static_cast<int>(max_confidence_male_uniform * 100);

       // Display the label and confidence near the top-left corner of the bounding box
       std::string label_with_confidence = male_uniform_label + " " + std::to_string(confidence_percent) + "%";
       putText(image, label_with_confidence, Point(max_confidence_male_rect.x, max_confidence_male_rect.y - 10), FONT_HERSHEY_SIMPLEX, 0.9, Scalar(0, 255, 0), 2);
   }
   else if (max_confidence_female_uniform > 0.0) {
       // Draw bounding box
       rectangle(image, max_confidence_female_rect, Scalar(255, 0, 0), 2);

       // Format the confidence as a percentage
       int confidence_percent = static_cast<int>(max_confidence_female_uniform * 100);

       // Display the label and confidence near the top-left corner of the bounding box
       std::string label_with_confidence = female_uniform_label + " " + std::to_string(confidence_percent) + "%";
       putText(image, label_with_confidence, Point(max_confidence_female_rect.x, max_confidence_female_rect.y - 10), FONT_HERSHEY_SIMPLEX, 0.9, Scalar(255, 0, 0), 2);
   }

   // Display the output image
   imshow("USTP Uniform Detection", image);
   waitKey(0);

   return 0;
}