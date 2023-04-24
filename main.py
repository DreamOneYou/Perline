import matplotlib.pyplot as plt
import numpy as np
import cv2
import matplotlib.animation as animation
from perlin_numpy import  generate_perlin_noise_3d, generate_fractal_noise_3d

def interpolant(t):
    return t*t*t*(t*(t*6 - 15) + 10)


def generate_perlin_noise_2d(
        shape, res, tileable=(False, False), interpolant=interpolant
):
    """生成2D perlin 噪声。

     参数：
         shape：生成的数组的形状（两个整数的元组）。
             这必须是 res 的倍数。
         res：沿每个方向产生的噪声周期数
             轴（两个整数的元组）。 音符形状必须是以下的倍数
             水库
         tileable：如果噪声应该沿每个轴平铺
             （两个布尔元组）。 默认为（假，假）。
         interpolant：插值函数，默认为
             t*t*t*(t*(t*6 - 15) + 10)。

     返回值：
         带有生成噪声的numpy数组。

     有待于解决问题：
         ValueError：如果 shape 不是 res 的倍数。
    """
    delta = (res[0] / shape[0], res[1] / shape[1])
    d = (shape[0] // res[0], shape[1] // res[1])
    grid = np.mgrid[0:res[0]:delta[0], 0:res[1]:delta[1]]\
             .transpose(1, 2, 0) % 1
    # Gradients
    angles = 2*np.pi*np.random.rand(res[0]+1, res[1]+1)
    gradients = np.dstack((np.cos(angles), np.sin(angles)))
    if tileable[0]:
        gradients[-1,:] = gradients[0,:]
    if tileable[1]:
        gradients[:,-1] = gradients[:,0]
    gradients = gradients.repeat(d[0], 0).repeat(d[1], 1)
    g00 = gradients[    :-d[0],    :-d[1]]
    g10 = gradients[d[0]:     ,    :-d[1]]
    g01 = gradients[    :-d[0],d[1]:     ]
    g11 = gradients[d[0]:     ,d[1]:     ]
    # Ramps
    n00 = np.sum(np.dstack((grid[:,:,0]  , grid[:,:,1]  )) * g00, 2)
    n10 = np.sum(np.dstack((grid[:,:,0]-1, grid[:,:,1]  )) * g10, 2)
    n01 = np.sum(np.dstack((grid[:,:,0]  , grid[:,:,1]-1)) * g01, 2)
    n11 = np.sum(np.dstack((grid[:,:,0]-1, grid[:,:,1]-1)) * g11, 2)
    # Interpolation
    t = interpolant(grid)
    c0 = n00*(1-t[:,:,0]) + t[:,:,0]*n10
    c1 = n01*(1-t[:,:,0]) + t[:,:,0]*n11
    return np.sqrt(2)*((1-t[:,:,1])*c0 + t[:,:,1]*c1)
def generate_fractal_noise_2d(
        shape, res, octaves=1, persistence=0.5,
        lacunarity=2, tileable=(False, False),
        interpolant=interpolant
):
    """生成分形噪声的 2D numpy 数组。

     参数：
         shape：生成的数组的形状（两个整数的元组）。 这必须是 lacunarity**(octaves-1)*res 的倍数。

        res：沿每个轴生成的噪声周期数（两个整数的元组）。 注意：形状必须是 (lacunarity**(octaves-1)*res) 的倍数。

        octaves：噪音中的八度音阶数。 默认为 1。
        persistence：两个八度音阶之间的比例因子。
        lacunarity：两个八度音阶之间的频率因子。
        tileable：如果噪音应该沿着每个轴平铺（两个布尔元组）。 默认为（假，假）。
        interpolant：插值函数，默认为t*t*t*(t*(t*6 - 15) + 10)。

    返回值：
         分形噪声的numpy数组和由perlin噪声结合的一些八音度数生成的numpy数组。

     有待于提高：
         ValueError：如果shape不是以下的倍数
            (lacunarity**(octaves-1)*res).。
    """
    noise = np.zeros(shape)
    frequency = 1
    amplitude = 1
    for _ in range(octaves):
        noise += amplitude * generate_perlin_noise_2d(
            shape, (frequency*res[0], frequency*res[1]), tileable, interpolant
        )
        frequency *= lacunarity
        amplitude *= persistence
    return noise

def perlin_2d_img(img):
    H, W = img.shape
    print(img.shape)
    img = img /255
    np.random.seed(0)
    noise = generate_perlin_noise_2d((H,W), (8,8))
    perline_noise = img + noise
    perline_noise = np.clip(perline_noise, 0, 1)
    perline_noise = np.uint8(perline_noise*255)
    plt.imshow(perline_noise,cmap="gray", interpolation="lanczos")
    plt.colorbar()

    np.random.seed(0)
    noise = generate_fractal_noise_2d((H,W), (8, 8), 5)
    plt.figure()
    perline_noise = img + noise
    perline_noise = np.clip(perline_noise, 0, 1)
    perline_noise = np.uint8(perline_noise * 255)
    plt.imshow(perline_noise, cmap="gray", interpolation="lanczos")
    plt.colorbar()
    plt.savefig(r"E:\myProject\my_tools\ICML\一些小code\Prelin噪声/img_perin.png")
    plt.show()
def perlin_2d():
    np.random.seed(0)
    noise = generate_perlin_noise_2d((256,256), (8,8))
    plt.imshow(noise,cmap="gray", interpolation="lanczos")
    plt.colorbar()

    np.random.seed(0)
    noise = generate_fractal_noise_2d((256, 256), (8, 8), 5)
    plt.figure()
    plt.imshow(noise, cmap="gray", interpolation="lanczos")
    plt.colorbar()
    plt.savefig(r"E:\myProject\my_tools\ICML\一些小code\Prelin噪声/generate_fractal_noise_2d.png")
    plt.show()
def Fractal_perlin_3D():
    np.random.seed(0)
    noise = generate_fractal_noise_3d(
        (32,256,256),
        (1, 4, 4),
        4,
        tileable=(True, False, False)
    )
    fig = plt.figure()
    images = [
        [plt.imshow(
            layer,
            cmap="Reds",
            interpolation="lanczos",
            animated=True)]
        for layer in noise
    ]
    animation_3d = animation.ArtistAnimation(fig, images, interval=50, blit=True)
    animation_3d.save(r"E:\myProject\my_tools\ICML\一些小code\Prelin噪声/Fractal_perlin_3D.gif",writer="pillow")
    plt.show()
def perline_3d():
    np.random.seed(0)
    noise = generate_perlin_noise_3d((32,256,256),
                                     (1,4,4),
                                     tileable=(True, False, False))
    fig = plt.figure()
    images = [
        [plt.imshow(layer, cmap="Reds", interpolation="lanczos",
                    animated=True)]
        for layer in noise
    ]
    animation_3d = animation.ArtistAnimation(fig, images,
                                             interval=50, blit=True)
    plt.show()
if __name__ == "__main__":
    img = cv2.imread("people.jpg", 0)
    # perlin_2d_img(img)  # 给图像添加perlin噪声
    # perlin_2d()  # 生成perlin噪声
    Fractal_perlin_3D()  # 生成3d perlin噪声
    # perline_3d()