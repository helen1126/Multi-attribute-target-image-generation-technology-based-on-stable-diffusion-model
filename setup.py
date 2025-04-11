from setuptools import setup, find_packages

setup(
    # ��Ŀ����
    name='Multi-attribute-target-image-generation',
    # ��Ŀ�汾
    version='0.1',
    # ��Ŀ����
    description='Research and implementation of multi-attribute target image generation technology based on stable diffusion model',
    # ��Ŀ��URL
    url='https://github.com/your_username/Multi-attribute-target-image-generation-technology-based-on-stable-diffusion-model',  # ���滻Ϊʵ����Ŀ�ֿ��ַ
    # �Զ�������Ŀ�е����а�
    packages=find_packages(),
    # ��Ŀ�����Python�汾
    python_requires='==3.10.*',
    # ��Ŀ��������
    install_requires=[
        'pytorch',
        'torchvision',
        'opencv-python',
        'Pillow',
        'matplotlib',
        'transformers',
        'tensorboard',
        'redis-py',
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
    # ��Ŀ�Ĺؼ���
    keywords='multi-attribute image generation, stable diffusion model',
)