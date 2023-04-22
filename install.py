import launch
print('hello from vidfaceswap')
if not launch.is_installed('torchgeometry'):
    launch.run_pip('install torchgeometry', 'requirements for geometry')
if not launch.is_installed('moviepy'):
    launch.run_pip('install moviepy', 'requirements for video i/o')
if not launch.is_installed('webp'):
    launch.run_pip('install webp', 'requirement for webp video format')
if not launch.is_installed('onnxruntime-gpu'):
    launch.run_pip('install onnxruntime-gpu', 'requirements for face detection using GPU')
if not launch.is_installed('insightface'):
    launch.run_pip('install -U insightface', 'requirements for face detection')