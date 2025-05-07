Get-ChildItem ".\torch_gpu\" | ForEach-Object { pip install $_.FullName }
