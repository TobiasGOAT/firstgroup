run:
	python3 -m fmnn25_project3.main

run_mpi_3:
	mpiexec -n 3 python3 -m fmnn25_project3.main

run_mpi_4:
	mpiexec -n 4 python3 -m fmnn25_project3.main --use-room4

run_without_mpi:
	python3 -m fmnn25_project3.without_mpi

format:
	black fmnn25_project3
	
check_lint:
	flake8 --max-line-length=100 fmnn25_project3
	pylint --disable=C0103,C0114,E0401 fmnn25_project3