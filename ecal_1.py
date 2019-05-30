import cv2
# import cv2.__version__[0]='3'
import numpy as np
import os
import glob

set_num=1

def main():
    DIM,K,D,R,T,pos=calibrate()
    # undistorted_img=undistort(image_path, DIM, K, D)
    # cv2.imwrtie('res_intrinsic.png', undistorted_img)

    # undistorted_img, new_K=undistort_dim(image_path, DIM, K, D, balance=0.7)
    # cv2.imwrite('res_intrinsic_balance.png', undistorted_img)

    img_path='./74cm/new/image002.png'
    img=cv2.imread(img_path)
    view_changed_image=change_camera_view(img, K, R, T, -pos[2])
    # cv2.imshow('change view',view_changed_image)
    # cv2.waitKey(100)
    cv2.imwrite('res.png', view_changed_image)

    print('DONE')

def calibrate():
    if set_num==1:
        CHECKBOARD=(9,6)
    elif set_num==2:
        CHECKBOARD =(5,6)
    else:
        CHECKBOARD=None

    subpix_criteria=(cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    # subpix_criteria=(cv2.TERM_CRITERIA_EPS+cv2.TERM_CRITERIA_ITER,30,0.1)
    objp=np.zeros((1,CHECKBOARD[0]*CHECKBOARD[1],3),np.float32)
    objp[0,:,:2]=np.mgrid[0:CHECKBOARD[0],0:CHECKBOARD[1]].T.reshape(-1,2)*2.0

    _img_shape=None
    objpoints=[]
    imgpoints=[]
    checkboard_path='./74cm/new/image002.png'
    images=glob.glob(checkboard_path)

    count = 0
    for frame in images:
        count+=1
        img=cv2.imread(frame)

        if _img_shape==None:
            _img_shape=img.shape[:2]
        # else:
            # assert _img_shape=img.shape[:2]
        gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

        # ret,corners=cv2.findChessboardCorners(gray,CHECKBOARD,cv2.CALIB_CB_ADAPTIVE_THRESH+cv2.CALIB_CB_FAST_CH+cv2.CALIB_CB_NORMALIZE_IMAGE)
        # ret,corners=cv2.findChessboardCorners(gray,CHECKBOARD,cv2.CALIB_CB_ADAPTIVE_THRESH+cv2.CALIB_CB_FAST_CHECK+cv2.CALIB_CB_NORMALIZE_IMAGE)
        ret,corners=cv2.findChessboardCorners(gray,CHECKBOARD,None)

        if ret==True:
            # cv2.imshow('img', img)
            # cv2.waitKey(500)
            objpoints.append(objp)
            cv2.cornerSubPix(gray,corners,(3,3),(-1,-1),subpix_criteria)
            # cv2.cornerSubpix(gray,corners,(3,3),(-1,-1),subpix_criteria)
            imgpoints.append(corners)
            img=cv2.drawChessboardCorners(img,CHECKBOARD,corners,ret)
            # cv2.imshow('img',img)
            # cv2.waitKey(500)
        cv2.imwrite('74cm/res/corners_'+str(count)+'.png',img)
        # cv2.imwrite('chessboard_corners.png',img)
    # cv2.destroyAllWindows()

    N_OK=len(objpoints)
    K=np.zeros((3,3))
    D=np.zeros((4,1))

    # retval,K,D,rvecs,tvecs=cv2.fisheye.calibrate(objpoints, imgpoints, gray.shape[::-1], None, D)
    retval,K,D,rvecs,tvecs=cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, D)
    retval,new_K,D,rvecs2,tvecs2=cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], K, D,flags=(cv2.CALIB_USE_INTRINSIC_GUESS + cv2.CALIB_FIX_PRINCIPAL_POINT))
    objpoints= np.array(objpoints).reshape(54, 1, 3)
    imgpoints= np.array(imgpoints).reshape(54, 1, 2)
    ret, rvec, tvec= cv2.solvePnP(objpoints, imgpoints, K,  np.zeros(5))
    # retval,new_K,D,rvecs,tvecs=cv2.calibrateCameraExtended(objpoints, imgpoints, gray.shape[::-1], K, D,flags=(cv2.CALIB_USE_INTRINSIC_GUESS + cv2.CALIB_FIX_PRINCIPAL_POINT))
    D = np.zeros((4, 1))

    R=cv2.Rodrigues(rvecs[0])
    T=tvecs

    R_inv=np.linalg.inv(R[0])
    pos=np.matmul(-R_inv,T[0])
    print("pos",pos)

    return _img_shape[::-1],new_K,D,R[0],T[0],pos
    # return _img_shape[::-1],K,D,R[0],T[-2],images[-2]

def undistort(img_path, DIM, K, D):
    img=cv2.imread(img_path)

    map1,map2=cv2.fisheye.initUndistortRectifyMap(K,D,np.eye(3),K,DIM,cv2.CV_16SC2)

    undistorted_img=cv2.remap(img,map1,map2,interpolation=cv2.INTER_LINEAR,borderMode=cv2.BORDER_CONSTATNT)

    cv2.imshow("undistorted",undistorted_img)
    cv2.waitKey(500)
    cv2.destroyAllWindows()

    return undistorted_img

def undistort_dim(img_path,DIM,K,D,balance=0,dim2=None,dim3=None):
    img=cv2.imread(img_path)
    dim1=img.shape [:2][:,:-1]

    if not dim2:
        dim2=dim1
    if not dim3:
        dim3=dim1

    scaled_K=K*dim1[0]/DIM[0]
    scaled_K[2][2]=1.0

    new_K=cv2.fisheye.estimateNewCameraMatrixForUndistortRectify(scaled_K,D,dim2,eye(3),balance=balance)

    map1,map2=cv2.fisheye.initUndistortRectifyMap(scaled_K,D,np.eye(3),new_K,DIM,cv2.CV_16SC2)
    undistorted_img=cv2.remap(img,map1,map2,interpolation=cv2.INTER_LINEAR,borderMoDE=cv2.BORDER_CONSTANT)

    return undistorted_img,new_K

def change_camera_view(image,camera_matrix,R,T,height=1):
    fx,_,xc=camera_matrix[0]
    _,fy,yc=camera_matrix[1]
    res=np.zeros(image.shape)

    for i in range(0,image.shape[0]):
        for j in range(0,image.shape[1]):
            xi=j
            yi=i

            xn=(xi-xc)/fx
            yn=(yi-yc)/fy

            a=np.array([xn,yn,1])*height
            aa=np.matmul(R,a)+np.transpose(T)

            xu=aa[0][0]/(aa[0][2]+0.000001)
            yu=aa[0][1]/(aa[0][2]+0.000001)
            # xu=aa[0]/(aa[2]+0.000001)
            # yu=aa[1]/(aa[2]+0.000001)

            xi=(xu*fx)+xc
            yi=(yu*fy)+yc

            if (xi>0)&(xi<image.shape[1]):
                if (yi>0)&(yi<image.shape[0]):
                    res[i,j,:]=image[int(yi),int(xi),:]

    return res


main()



