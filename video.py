import cv2
import pytesseract

# Path to the Tesseract executable
pytesseract.pytesseract.tesseract_cmd = r"C:/Program Files/Tesseract-OCR/tesseract.exe"

# Initialize video capture
cap = cv2.VideoCapture(0)  # Use 0 for the default camera

try:
    while True:
        # Read the frame from the camera
        ret, img = cap.read()
        if not ret:
            print("Failed to grab frame")
            break
        
        # Convert image to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Apply bilateral filter
        filter_img = cv2.bilateralFilter(gray, 11, 15, 15)
        
        # Edge detection
        edge = cv2.Canny(filter_img, 170, 200)
        
        # Contour detection
        contours, _ = cv2.findContours(edge, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        
        # Sort contours by area and find the largest rectangle
        contours = sorted(contours, key=cv2.contourArea, reverse=True)[:10]
        screenCnt = None
        
        for c in contours:
            peri = cv2.arcLength(c, True)
            approx = cv2.approxPolyDP(c, 0.018 * peri, True)
            
            if len(approx) == 4:
                screenCnt = approx
                x, y, w, h = cv2.boundingRect(approx)
                img2 = img[y:y+h, x:x+w]
                
                # OCR to read text from the number plate
                config = '-l eng --oem 1 --psm 3'
                text = pytesseract.image_to_string(img2, config=config)
                print(text)
                
                # Draw the contours on the image
                cv2.drawContours(img, [screenCnt], -1, (255, 0, 0), 3)
                break
        
        # Display the result
        cv2.imshow("Number Plate Recognition", img)
        
        # Break the loop when 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
finally:
    # Release resources
    cap.release()
    cv2.destroyAllWindows()
