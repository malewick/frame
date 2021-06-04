; The name of the installer
Name "FRAME Installation Wizard"

; The file to write
OutFile "setupFRAME.exe"

; The default installation directory
InstallDir $DESKTOP\FRAME

; The text to prompt the user to enter a directory
DirText "This will install FRAME on your computer. Choose a directory"

RequestExecutionLevel user

;--------------------------------
Section ""
SetShellVarContext current

; Set output path to the installation directory.
SetOutPath $INSTDIR

; Put file there
File /r "c:\Users\Test\Desktop\frame\dist\FRAME"
File /r "c:\Users\Test\Desktop\frame\dist\input"

# Start Menu
createDirectory "$SMPROGRAMS\FRAME"
createShortCut "$SMPROGRAMS\FRAME\FRAME.lnk" "$INSTDIR\FRAME\FRAME.exe" "" ""

# .. directory
createShortCut "$INSTDIR\FRAME.lnk" "$INSTDIR\FRAME\FRAME.exe" "" ""

# Desktop
createShortCut "$DESKTOP\FRAME.lnk" "$INSTDIR\FRAME\FRAME.exe" "" ""

# define uninstaller name
WriteUninstaller $INSTDIR\uninstaller.exe

SectionEnd


Section "Uninstall"

Delete $INSTDIR\uninstaller.exe
Delete $INSTDIR\*
RMDir /r $INSTDIR
Delete "$SMPROGRAMS\FRAME\FRAME.lnk"
RMDir /r "$SMPROGRAMS\FRAME"
Delete "$DESKTOP\FRAME.lnk"
Delete "$DESKTOP\FRAME.lnk"

SectionEnd
