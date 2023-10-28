import cv2
import mediapipe as mp

mp_hands = mp.solutions.hands
hands = mp_hands.Hands()

def detect_fingers():
    cap = cv2.VideoCapture(0)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            continue

        # Convert the frame to RGB for processing with mediapipe
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Process the frame with mediapipe hands
        results = hands.process(frame_rgb)

        if results.multi_hand_landmarks:
            for landmarks in results.multi_hand_landmarks:
                finger_count = 0

                # Detect thumb (landmark points 0-4)
                if landmarks.landmark[4].x < landmarks.landmark[3].x:
                    finger_count += 1

                # Detect index finger (landmark points 5-8)
                if landmarks.landmark[8].y < landmarks.landmark[5].y:
                    finger_count += 1

                # Detect middle finger (landmark points 9-12)
                if landmarks.landmark[12].y < landmarks.landmark[9].y:
                    finger_count += 1

                # Detect ring finger (landmark points 13-16)
                if landmarks.landmark[16].y < landmarks.landmark[13].y:
                    finger_count += 1

                # Detect little finger (landmark points 17-20)
                if landmarks.landmark[20].y < landmarks.landmark[17].y:
                    finger_count += 1

                cv2.putText(frame, f'Finger Count: {finger_count}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

            mp_drawing = mp.solutions.drawing_utils
            mp_drawing.draw_landmarks(frame, landmarks, mp_hands.HAND_CONNECTIONS)
        cv2.imshow('Hand Gesture Detection', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    detect_fingers()
