import mtcnn
import cv2 , os , glob
import numpy as np

detector = mtcnn.MTCNN()

players_path = "D:\Dataset\Arsenal_players/"
Outpath = "images/"
folders = os.listdir(players_path)

for i, player in enumerate(folders):
    print("\rProcessing ... {}/{}".format(i+1,len(folders)) , end="")
    out_folder = Outpath + player

    if not os.path.isdir(out_folder):
         os.makedirs(out_folder)
    else:
        continue
    images = glob.glob(players_path + player +"\\*.jpg")
    for image in images:
        outname = image.split("\\")[-1]
        flag = False
        img = cv2.imread(image)
        if img is None:
            img = cv2.imdecode(np.fromfile(image, dtype=np.uint8), cv2.IMREAD_UNCHANGED)
            flag = True
        try:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        except:
            print("Failed! - " +image)
            continue
        dets = detector.detect_faces(img)
        if len(dets) != 1 : continue
        for i, d in enumerate(dets):
            x1, y1, w, h =d['box']

            xc, yc = int(x1+w/2), int(y1+h/2) 
            if w > h : h = w
            else: w = h

            x1 ,y1 , x2, y2 = int(xc - w/2) ,int(yc -h/2), int(xc + w/2), int(yc+ h/2)
            try:
                cropface = img[y1:y2 , x1:x2]
                cropface = cv2.cvtColor(cropface, cv2.COLOR_BGR2RGB)
                if cropface.shape[0] < 50 or cropface.shape[1] < 50: continue
                if not flag:
                    cv2.imwrite(out_folder+"/"+outname,cropface)
                else:
                    cv2.imencode(".jpg",cropface)[1].tofile(out_folder+"/"+outname)
            except:
                print("Except!")
                continue