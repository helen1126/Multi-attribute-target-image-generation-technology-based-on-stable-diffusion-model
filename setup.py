from setuptools import setup, find_packages

setup(
    # 项目名称
    name='Multi-attribute-target-image-generation',
    # 项目版本
    version='0.1',
    # 项目描述
    description='Research and implementation of multi-attribute target image generation technology based on stable diffusion model',
    # 项目的URL
    url='https://github.com/helen1126/Multi-attribute-target-image-generation-technology-based-on-stable-diffusion-model',
    # 自动发现项目中的所有包
    packages=find_packages(),
    # 项目所需的Python版本
    python_requires='==3.10.*',
    # 项目的依赖项
    install_requires=[
        'torch',
        'torchvision',
        'opencv-python',
        'Pillow',
        'matplotlib',
        'transformers',
        'tensorboard',
        'redis',
        'geoip2',
        'flask',
        'scikit-learn',
        'numpy',
        'requests',
        'openai',
        'GPUtil',
        'openai-clip',
        'user-agents'
    ],
    # 项目的关键字
    keywords='multi-attribute image generation, stable diffusion model',
)
