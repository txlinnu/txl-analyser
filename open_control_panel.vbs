' Desktop-icon launcher for the TXL Control Panel.
' If the panel isn't already running, starts it silently (no console window),
' then opens it in your default browser either way.

Dim baseDir, pythonw, panelUrl
baseDir = "D:\AI Agents"
pythonw = "C:\Users\pc\AppData\Local\Programs\Python\Python312\pythonw.exe"
panelUrl = "http://127.0.0.1:5099"

Function IsUp(url)
    On Error Resume Next
    Dim http
    Set http = CreateObject("MSXML2.XMLHTTP")
    http.Open "GET", url, False
    http.Send
    If Err.Number = 0 And http.Status >= 200 And http.Status < 500 Then
        IsUp = True
    Else
        IsUp = False
    End If
    On Error Goto 0
End Function

Dim shell
Set shell = CreateObject("WScript.Shell")

If Not IsUp(panelUrl & "/api/status") Then
    shell.CurrentDirectory = baseDir
    shell.Run """" & pythonw & """ """ & baseDir & "\control_panel.py""", 0, False
    WScript.Sleep 2000
End If

shell.Run panelUrl, 1, False
