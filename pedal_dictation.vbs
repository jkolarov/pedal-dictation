Set WshShell = CreateObject("WScript.Shell")
Set fso = CreateObject("Scripting.FileSystemObject")
scriptDir = fso.GetParentFolderName(WScript.ScriptFullName)

' Locate python.exe via registry (NOT pythonw.exe).
' pythonw.exe uses CREATE_NO_WINDOW which prevents pynput from initialising
' its hidden message window for keyboard hooks. python.exe with windowStyle=0
' creates a hidden console that pynput can use, with no window visible to user.
Dim python, versions, i, installPath
python = "python"
versions = Array("3.14", "3.13", "3.12", "3.11", "3.10", "3.9")

On Error Resume Next
For i = 0 To UBound(versions)
    installPath = WshShell.RegRead("HKLM\SOFTWARE\Python\PythonCore\" & versions(i) & "\InstallPath\")
    If installPath <> "" Then
        If fso.FileExists(installPath & "python.exe") Then
            python = installPath & "python.exe"
            Exit For
        End If
    End If
Next
On Error GoTo 0

Dim q
q = Chr(34)
WshShell.Run q & python & q & " " & q & scriptDir & "\pedal_dictation.py" & q, 0, False
