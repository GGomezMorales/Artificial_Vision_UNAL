c = get_config()

c.NotebookApp.allow_origin = '*'
c.NotebookApp.ip = '0.0.0.0'
c.NotebookApp.port = 8888
c.NotebookApp.open_browser = True
c.ServerApp.allow_remote_access = True
c.ServerApp.open_browser = False
c.ServerApp.local_hostnames = ['localhost']
c.ServerApp.file_to_run = 'artificial-vision-project/main.ipynb'
c.ServerApp.kernel_manager_class = 'jupyter_server.services.kernels.kernelmanager.AsyncMappingKernelManager'
c.NotebookApp.token = ''
c.NotebookApp.password = ''
