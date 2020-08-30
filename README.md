# AnalysevonSensordaten
~~ Der Code Upload für unsere Hausarbeit ~~

Der Ordner "wavfiles" beinhaltet die unbearbeiteten "[].wav" Datein. Diese werden von clean.py verwendet. 
In dem Ordner "clean" sind dann die Neuen, bearbeiteten Datein. Durch das Aufteilen einer relativ großen Datei in viele Kleine entstehen mehrmals so viele Sampels.
Im Ordner Models werden die drei Modelle als einzelne Datei abgespeichert. Dadurch konnten wir sie später in predict.py wiederverwenden.
Die zwei Jupyter Notebooks haben wir verwendet, um die Bilder für unsere Präsentation zu besorgen.

Bevor man den Code selbst ausführen will, müssen einige Anpassungen gemacht werden. 
Es gibt mehrmals in den Datein die Pfade für das Aufrufen der Wav-Files. 
Ganz unten in dem Code, findet man Vereinfachungen für Eingaben durch das Package "Argparse". Dort sollte man seine Anpassungen eingeben und die werden so im gesamten Code angepasst. 

Ablauf vom Abrufen der einzelnen "[].py" Datein zum selbst ausprobieren: 
1. Clean.py ist die Datei, welche unsere Wav-Files sauber machen: Länge anpassen, downsamplen usw..

2. train.py ist die Datei, welche unsere Modelle kreirt und dazu das trainierte Modell dahinter abspeichert, um dieses später wiederzuverwenden.
  2.1 model.py wird von train.py aufgerufen. Hier sind die einzelnen Modelle hinterlegt. 

3. Predict.py ist die Datei, welche unsere abgespeicherten Modelle aufruft, um zu testen, wie gut eigentlich unsere Modelle sind auf nicht trainierte Datensätze.
