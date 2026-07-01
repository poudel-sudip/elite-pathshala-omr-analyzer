import cv2
import numpy as np
import json

def locate_page_corner_markers(image_path, debug=False):

    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Could not open or find the image: {image_path}")
            
    h, w, _ = img.shape

    # print("\n Width:"+str(w))
    # print("\n Height:"+str(h))

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 180, 255, cv2.THRESH_BINARY_INV)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    candidate_centers = []

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < 100 or area > 6000:
            continue

        x_box, y_box, w_box, h_box = cv2.boundingRect(cnt)
        aspect_ratio = float(w_box) / h_box

        if 0.2 <= aspect_ratio <= 4.5:
            M = cv2.moments(cnt)
            if M["m00"] != 0:
                cX = int(M["m10"] / M["m00"])
                cY = int(M["m01"] / M["m00"])
                candidate_centers.append((cX, cY))


    if len(candidate_centers) < 4:
        raise ValueError(f"Found only {len(candidate_centers)} candidate markers. Need at least 4.")

    pts = np.array(candidate_centers)
    sum_pts = pts.sum(axis=1)
    diff_pts = np.diff(pts, axis=1).flatten()
    
    tl = pts[np.argmin(sum_pts)]
    br = pts[np.argmax(sum_pts)]
    tr = pts[np.argmin(diff_pts)]
    bl = pts[np.argmax(diff_pts)]

    corner_coordinates = {
        "tl":     {"x": int(tl[0]), "y": int(tl[1])},
        "tr":    {"x": int(tr[0]), "y": int(tr[1])},
        "br": {"x": int(br[0]), "y": int(br[1])},
        "bl":  {"x": int(bl[0]), "y": int(bl[1])}
    }

    with open("template_corner_markers.json", "w") as f:
        json.dump(corner_coordinates, f, indent=4)

    if debug:
        vis_img = img.copy()
        sorted_corners = np.array([tl, tr, br, bl], dtype="float32")
        labels = ["TL", "TR", "BR", "BL"]
        colors = [(0, 0, 255), (0, 255, 0), (255, 0, 0), (0, 125, 255)]

        for idx, (x, y) in enumerate(sorted_corners):
            pt = (int(x), int(y))
            cv2.circle(vis_img, pt, 25, colors[idx], -1)
            cv2.putText(vis_img, labels[idx], (pt[0] + 30, pt[1] + 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1.2, colors[idx], 4)
            # print(f"{labels[idx]} Corner Found at Coordinate: X={pt[0]}, Y={pt[1]}")
            
        cv2.imwrite("template_corner_markers.jpg", vis_img)

    return corner_coordinates

def locate_student_id_bubbles(image_path, debug=False):

    COLS = 8
    ROWS = 10

    TL = np.array([1620, 1490])
    TR = np.array([2260, 1490])

    BL = np.array([1620, 2015])
    BR = np.array([2260, 2015])

    COL_CORRECTIONS = {
        0: 0,   
        1: 0,   
        2: -10,
        3: -8,
        4: -7,
        5: -5,
        6: -5,
        7: 0 
    }

    ROW_CORRECTIONS = {
        0: 0,   
        1: 0,   
        2: -2,
        3: -5,
        4: -5,
        5: -8,
        6: -5,
        7: -4,
        8: -4,
        9: 0
    }

    coords = {}

    for col in range(COLS):

        top = TL + ((TR - TL) * col / (COLS - 1))
        bottom = BL + ((BR - BL) * col / (COLS - 1))

        for row in range(ROWS):

            p = top + ((bottom - top) * row / (ROWS - 1))
            
            corrected_x = int(round(p[0])) + COL_CORRECTIONS.get(col, 0)
            corrected_y = int(round(p[1])) + ROW_CORRECTIONS.get(row, 0)

            # corrected_x = int(round(p[0]))
            # corrected_y = int(round(p[1]))

            coords[f"{col}_{row}"] = {
                "x": corrected_x,
                "y": corrected_y
            }

    # Save the perfectly aligned template coordinates
    with open("template_student_id_bubbles.json", "w") as f:
        json.dump(coords, f, indent=4)

    if debug:
        img = cv2.imread(image_path)
        if img is None:
            raise FileNotFoundError(f"Could not open or find the image: {image_path}")
        
        with open("template_student_id_bubbles.json") as f:
            coords = json.load(f)

        for key, pt in coords.items():
            cv2.circle(
                img,
                (pt["x"], pt["y"]),
                8,
                (255, 0, 0),
                -1
            )
        
        cv2.imwrite("template_student_id_bubbles.jpg", img)
        

    return coords

def locate_questions_bubbles(image_path, debug=False):
    BLOCKS = 5
    QUESTIONS_PER_BLOCK = 10
    OPTIONS = 4

    BLOCK_CORNERS = {
        "1" : {
            "TL" : np.array([202, 2263]),
            "TR" : np.array([490, 2263]),
            "BR" : np.array([485, 3204]),
            "BL" : np.array([198, 3205])
        },
        "2" : {
            "TL" : np.array([660, 2265]),
            "TR" : np.array([948, 2266]),
            "BR" : np.array([945, 3207]),
            "BL" : np.array([658, 3206])
        },
        "3" : {
            "TL" : np.array([1113, 2272]),
            "TR" : np.array([1400, 2272]),
            "BR" : np.array([1400, 3215]),
            "BL" : np.array([1110, 3215])
        },
        "4" : {
            "TL" : np.array([1572, 2270]),
            "TR" : np.array([1860, 2271]),
            "BR" : np.array([1858, 3211]),
            "BL" : np.array([1570, 3212])
        },
        "5" : {
            "TL" : np.array([2023, 2272]),
            "TR" : np.array([2310, 2272]),
            "BR" : np.array([2310, 3215]),
            "BL" : np.array([2022, 3215])
        },
    }

    COL_CORRECTIONS = {
        1 : -5,
        2 : -5
    }

    ROW_CORRECTIONS = {
        "1_3" : -10,
        "1_6" : 2,
        "1_7" : 5,
        "1_8" : 7,
        "2_2" : 1,
        "2_3" : -4,
        "2_4" : -8,
        "2_6" : 5,
        "2_7" : 9,
        "2_8" : 10,
        "3_1" : -6,
        "3_2" : -2,
        "3_3" : -8,
        "3_4" : -10,
        "3_6" : -5,
        "3_8" : 3,
        "4_1" : -4,
        "4_2" : -4,
        "4_3" : -10,
        "4_4" : -8,
        "4_5" : -2,
        "4_7" : 4,
        "4_8" : 1,
        "5_1" : -3,
        "5_3" : -6,
        "5_4" : -13,
        "5_5" : -2,
        "5_6" : -5,
        "5_8" : 5,
    }
    
    coords = {}
    question = 1

    for block in range(BLOCKS):        
        for row in range(QUESTIONS_PER_BLOCK):
            left = BLOCK_CORNERS[str(block+1)]["TL"] + ((BLOCK_CORNERS[str(block+1)]["BL"] - BLOCK_CORNERS[str(block+1)]["TL"]) * row / (QUESTIONS_PER_BLOCK - 1))
            right = BLOCK_CORNERS[str(block+1)]["TR"] + ((BLOCK_CORNERS[str(block+1)]["BR"] - BLOCK_CORNERS[str(block+1)]["TR"]) * row / (QUESTIONS_PER_BLOCK - 1))
            
            coords[str(question)] = {}
            for col, option in enumerate(["A","B","C","D"]):
                p = left + ((right - left) * col / (OPTIONS - 1))

                cx = int(round(p[0])) + COL_CORRECTIONS.get(col, 0)
                cy = int(round(p[1])) + ROW_CORRECTIONS.get(f"{block+1}_{row}", 0)

                coords[str(question)][option] = {
                    "x": cx,
                    "y": cy
                }

            question += 1

    with open("template_question_bubbles.json", "w") as f:
        json.dump(coords, f, indent=4)

    if debug:
        img = cv2.imread(image_path)
        if img is None:
            raise FileNotFoundError(f"Could not open or find the image: {image_path}")
        
        with open("template_question_bubbles.json") as f:
            coords = json.load(f)
        
        for q, options in coords.items():
            for opt, pt in options.items():
                cv2.circle(img,
                    (
                        pt["x"],
                        pt["y"]
                    ),
                    8,
                    (255,0,0),
                    -1
                )

                cv2.putText(img,
                    f"{q}{opt}",
                    (
                        pt["x"]-10,
                        pt["y"]-10
                    ),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.35,
                    (0,0,255),
                    1
                )

                cv2.imwrite("template_question_bubbles.jpg",img)

    return coords

def locate_set_key_bubbles(image_path, debug=False):
         
    BLOCK_COORDS = {
        "A" : np.array([1840, 2137]),
        "B" : np.array([1931, 2137]),
        "C" : np.array([2026, 2137]),
        "D" : np.array([2127, 2137])
    }

    coords = {}
    for col, option in enumerate(["A","B","C","D"]):
        p = BLOCK_COORDS[str(option)]
        cx = int(round(p[0]))
        cy = int(round(p[1]))

        coords[str(option)] = {
            "x": cx,
            "y": cy
        }

    with open("template_set_key_bubbles.json", "w") as f:
        json.dump(coords, f, indent=4)
   

    if debug:
        img = cv2.imread(image_path)
        if img is None:
            raise FileNotFoundError(f"Could not open or find the image: {image_path}")
        
        with open("template_set_key_bubbles.json") as f:
            coords = json.load(f)
        
        for opt, pt in coords.items():
            cv2.circle(img,
                (
                    pt["x"],
                    pt["y"]
                ),
                8,
                (255,0,0),
                -1
            )

            cv2.putText(img,
                f"{opt}",
                (
                    pt["x"]-10,
                    pt["y"]-10
                ),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.35,
                (0,0,255),
                1
            )

            cv2.imwrite("template_set_key_bubbles.jpg",img)

    return coords
 

def generate_merged_debug_image(image_path):
    page_corners = {}
    student_id_bubbles = {}
    set_bubbles = {}
    questions_bubbles = {}

    with open("template_corner_markers.json") as f:
        page_corners = json.load(f)
    
    with open("template_student_id_bubbles.json") as f:
        student_id_bubbles = json.load(f)
    
    with open("template_set_key_bubbles.json") as f:
        set_bubbles = json.load(f)
        
    with open("template_question_bubbles.json") as f:
        questions_bubbles = json.load(f)


    # 1. Read base image once
    vis_img = cv2.imread(image_path)
    if vis_img is None:
        raise FileNotFoundError(f"Could not open or find the image for debugging: {image_path}")

    # 2. Draw Page Corner Markers
    labels = ["TL", "TR", "BR", "BL"]
    colors = [(0, 0, 255), (0, 255, 0), (255, 0, 0), (0, 125, 255)] # BGR
    sorted_keys = ["tl", "tr", "br", "bl"]
    
    for idx, key in enumerate(sorted_keys):
        pt = (page_corners[key]["x"], page_corners[key]["y"])
        cv2.circle(vis_img, pt, 25, colors[idx], -1)
        cv2.putText(vis_img, labels[idx], (pt[0] + 30, pt[1] + 10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, colors[idx], 4)

    # 3. Draw Student ID Bubbles (Blue dots)
    for key, pt in student_id_bubbles.items():
        cv2.circle(vis_img, (pt["x"], pt["y"]), 8, (255, 0, 0), -1)

    # 4. Draw Set Key Bubbles (Blue dots with red labels)
    for key, pt in set_bubbles.items():
        cv2.circle(vis_img, (pt["x"], pt["y"]), 8, (255, 0, 0), -1)
        cv2.putText(vis_img, f"{key}", (pt["x"] - 10, pt["y"] - 10),
            cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 255), 1)

    # 5. Draw Question Bubbles (Blue dots with red labels)
    for q, options in questions_bubbles.items():
        for opt, pt in options.items():
            cv2.circle(vis_img, (pt["x"], pt["y"]), 8, (255, 0, 0), -1)
            cv2.putText(vis_img, f"{q}{opt}", (pt["x"] - 10, pt["y"] - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 255), 1)

    # 6. Save the combined master view
    cv2.imwrite("template_marked_image.jpg", vis_img)
    
    return


# Run detection on your template file
try:
    image_file = "template.jpg"
    # page_corners = locate_page_corner_markers(image_file, False)
    # student_id_bubbles = locate_student_id_bubbles(image_file, False)
    # set_key_bubbles = locate_set_key_bubbles(image_file, False)
    # questions_bubbles = locate_questions_bubbles(image_file, False)

    merged_image = generate_merged_debug_image(image_file)

except Exception as e:
    print(f"Error processing image: {e}")

