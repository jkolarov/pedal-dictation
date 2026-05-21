Set WshShell = CreateObject("WScript.Shell")
Set fso = CreateObject("Scripting.FileSystemObject")
scriptDir = fso.GetParentFolderName(WScript.ScriptFullName)

' Locate pythonw.exe via registry so this works at startup (before user PATH loads)
' and without hardcoding a version-specific path.
Dim pythonw, versions, i, installPath
pythonw = "pythonw"
versions = Array("3.14", "3.13", "3.12", "3.11", "3.10", "3.9")

On Error Resume Next
For i = 0 To UBound(versions)
    installPath = WshShell.RegRead("HKLM\SOFTWARE\Python\PythonCore\" & versions(i) & "\InstallPath\")
    If installPath <> "" Then
        If fso.FileExists(installPath & "pythonw.exe") Then
            pythonw = installPath & "pythonw.exe"
            Exit For
        End If
    End If
Next
On Error GoTo 0

WshShell.Run """" & pythonw & """ """ & scriptDir & "\pedal_dictation.py""", 0, False
