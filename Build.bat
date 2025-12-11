@echo off

REM default options
set USE_SSE=ON
set BUILD_TESTS=ON
set BUILD_PLAYGROUND=ON
set CMAKE_GENERATOR=Visual Studio 17 2022
set CMAKE_ARCH=x64

if exist BuildConfig.cfg (
	echo Using BuildConfig.cfg
	
	REM call  BuildConfig.cfg
	for /F "usebackq tokens=1,2 delims== eol=#" %%A in ("BuildConfig.cfg") do (
		set %%A=%%B
	)
	
	echo Generating build with:
) else (
	echo No Build Config
	echo Generating build with default:
)

echo.
echo USE_SSE = %USE_SSE%
echo BUILD_TESTS = %BUILD_TESTS%
echo BUILD_PLAYGROUND = %BUILD_PLAYGROUND%
echo CMAKE_GENERATOR = %CMAKE_GENERATOR%
echo CMAKE_ARCH = %CMAKE_ARCH%
echo.

if "%CMAKE_GENERATOR%"=="Visual Studio 17 2022" (
	echo Using VS 2022
) else if "%CMAKE_GENERATOR%"=="Visual Studio 16 2019" (
	echo Using VS 2019
) else (
	echo WARNING: Unknown generator %CMAKE_GENERATOR%
)

echo.
cmake -S . -B Build ^
	-DUSE_SSE=%USE_SSE% ^
	-DBUILD_TESTS=%BUILD_TESTS% ^
	-DBUILD_PLAYGROUND=%BUILD_PLAYGROUND% ^
	-G "%CMAKE_GENERATOR%" -A %CMAKE_ARCH%

echo.
echo Build Completed.
pause