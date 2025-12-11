@echo off

REM Delete the entire build folder
if exist Build (
	echo Deleting Build folder...
	rmdir /s /q Build
) else (
	echo No Build folder found.
)


echo Clean Completed.
pause