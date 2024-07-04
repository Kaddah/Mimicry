# Mimicry

Projekt erstellt für das WPF Visual Computing BaBs.
Der Zuschauer soll sich in den Werken bekannter Künstler wiederfinden.<br />
## Storyboard 
<br />

>In Coburg gibt es einen alten Antiquitätenladen. In diesem gibt es eine Vielzahl an alten, aber hübschen Möbeln und Kunstwerken.
>Der Besitzer, ein mysteriöser junger Mann zeigt dir einen Spiegel, mit dem du in verschiedene Welten eintauchen kannst. 
>Wenn du länger in den Spiegel schaust, merkst du, wie sich der Laden um dich herum verändert.<br />
>Plötzlich findest du dich in einer Welt, die du kennst, das sieht aus wie ein Bild, das du schonmal im Museum gesehen hast.<br />
>Wenn du dich selbst anschaust erkennst du dich auch nicht wieder. Du siehst so aus, als wärst du ein Teil des Bildes.
>Direkt aus dem Pinsel des Künstlers entstanden.

## Installationsanleitung

Die verwendeten Bibliotheken sind in der Datei 'requirements.txt' hinterlegt. 

Diese kann über den folgenden Befehl installiert werden:


```bash
pip install -r requirements.txt
```

>Die Python-Version muss zwischen 3.8 und 3.11 liegen. <br>
>Auf macOS läuft mediapipe auf einer niedrigeren Version als 0.10 (z.B. 0.9).

## Programm bauen

PyInstaller wird benötigt und kann mit pip installiert werden:

```bash
pip install pyinstaller
```

Unter Windows kann die .exe-Datei dann mit folgendem Kommando gebaut werden:

```bash
python -m PyInstaller main.py
``` 

Anschließend müssen einige Dateien manuell in das Bauverzeichnis kopiert werden.

Es werden die Module von mediapipe benötigt. Mit dem folgenden Befehl kann der Pfad der mediapipe-Bibliothek angezeigt werden:

```bash 
python -m pip show mediapipe
```

Im mediapipe-Ordner muss der modules-Ordner kopiert und im dist-Ordner unter main/_internal/mediapipe abgelegt werden. 
Weiterhin werden die Ordner `images` und `assets` benötigt. Diese gehören ebenfalls in den Ordner dist/main.