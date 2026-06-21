import cv2

SCALE = 0.3  # adjust if needed

points = []

img = cv2.imread("samples/empty.jpg")

h, w = img.shape[:2]

display = cv2.resize(
    img,
    (int(w * SCALE), int(h * SCALE))
)

clone = display.copy()


def click(event, x, y, flags, param):

    if event == cv2.EVENT_LBUTTONDOWN:

        # Convert display coordinates back to original image coordinates
        real_x = int(x / SCALE)
        real_y = int(y / SCALE)

        points.append((real_x, real_y))

        cv2.circle(clone, (x, y), 5, (0, 0, 255), -1)

        cv2.putText(
            clone,
            str(len(points)),
            (x + 10, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 0, 0),
            2
        )


cv2.namedWindow("template")
cv2.setMouseCallback("template", click)

while True:

    cv2.imshow("template", clone)

    key = cv2.waitKey(1)

    if key == 27:
        break

cv2.destroyAllWindows()

print(points)