
import cv2
import numpy as np


def dark_channel(img, window_size=15):
    """
    计算暗通道
    img: RGB, [0,1]
    """
    min_channel = np.min(img, axis=2)

    kernel = cv2.getStructuringElement(
        cv2.MORPH_RECT,
        (window_size, window_size)
    )

    dark = cv2.erode(
        min_channel,
        kernel
    )

    return dark


def estimate_atmospheric_light(img, dark):
    """
    估计大气光 A
    """

    h, w = dark.shape

    num_pixels = max(int(h*w*0.001), 1)

    dark_flat = dark.reshape(-1)

    img_flat = img.reshape(-1, 3)


    indices = np.argsort(dark_flat)[-num_pixels:]


    brightest = img_flat[indices]


    A = np.mean(
        brightest,
        axis=0
    )

    return A



def estimate_transmission(img, A, omega=0.95, window_size=15):
    """
    估计透射率
    """

    normalized = img / A.reshape(1,1,3)

    dark = dark_channel(
        normalized,
        window_size
    )


    t = 1 - omega * dark

    return t



def guided_filter(I, p, radius=40, eps=1e-3):
    """
    引导滤波
    """
    mean_I = cv2.boxFilter(
        I,
        cv2.CV_64F,
        (radius,radius)
    )

    mean_p = cv2.boxFilter(
        p,
        cv2.CV_64F,
        (radius,radius)
    )

    corr_I = cv2.boxFilter(
        I*I,
        cv2.CV_64F,
        (radius,radius)
    )

    corr_Ip = cv2.boxFilter(
        I*p,
        cv2.CV_64F,
        (radius,radius)
    )


    var_I = corr_I - mean_I*mean_I

    cov_Ip = corr_Ip - mean_I*mean_p


    a = cov_Ip/(var_I+eps)

    b = mean_p-a*mean_I


    mean_a = cv2.boxFilter(
        a,
        cv2.CV_64F,
        (radius,radius)
    )

    mean_b = cv2.boxFilter(
        b,
        cv2.CV_64F,
        (radius,radius)
    )


    q = mean_a*I+mean_b

    return q



def recover(img, t, A, t0=0.1):

    J = np.empty_like(img)

    for c in range(3):
        J[:,:,c] = (
            (img[:,:,c]-A[c])
            /
            np.maximum(t,t0)
            +
            A[c]
        )

    return np.clip(J,0,1)



def dehaze(input_path, output_path):

    img = cv2.imread(
        input_path
    )

    img = cv2.cvtColor(
        img,
        cv2.COLOR_BGR2RGB
    )

    img = img.astype(np.float32)/255.


    # dark channel
    dark = dark_channel(
        img
    )


    # atmospheric light
    A = estimate_atmospheric_light(
        img,
        dark
    )

    print("Atmospheric light:", A)


    # transmission
    t = estimate_transmission(
        img,
        A
    )


    # refine transmission
    gray = cv2.cvtColor(
        (img*255).astype(np.uint8),
        cv2.COLOR_RGB2GRAY
    )

    gray = gray.astype(np.float32)/255.

    t = guided_filter(
        gray,
        t
    )


    # recover
    result = recover(
        img,
        t,
        A
    )


    result = (
        result*255
    ).astype(np.uint8)


    result = cv2.cvtColor(
        result,
        cv2.COLOR_RGB2BGR
    )


    cv2.imwrite(
        output_path,
        result
    )


if __name__ == "__main__":

    dehaze(
        "035.png",
        "035dehazed.png"
    )