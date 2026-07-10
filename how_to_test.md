# Regeln für automatisierte Tests

## Unit-Tests

### Symmetrie

Für Unit-Tests gilt:

> Test-Code soll symmetrisch zum Production-Code sein.

Die Symmetrie ist im Regelfall eine *1:1-Zuordnung* zwischen relevanten öffentlichen Einheiten des Production-Codes und ihren Tests.

#### Vorteile

Diese Symmetrie hat folgende Vorteile:

- *Mentale Entlastung*: Das mentale Modell für den Production-Code kann für den Test-Code wiederverwendet werden.
- *Leichtes Finden*: Man findet leicht den Test zum Production-Code und umgekehrt.
- *Klarere Testabdeckung*: Es wird leichter klar, inwieweit die API durch Tests abgedeckt ist.
- *Leichtere Wartbarkeit*: Bei Umbenennung oder Verschiebung von Production-Code (Refactoring) weiß man sofort, welche Tests mitgezogen werden müssen.

#### Details

Konkret bedeutet die Symmetrie:

- *Test Module per Module*: Zu einem Production-Modul gibt es ein korrespondierendes Test-Modul.[^1]
- Der Name des Test-Moduls entspricht dem Namen des Production-Moduls, z. B. `foo.py` und `test_foo.py`.
- Zu jeder relevanten öffentlichen Funktion gibt es mindestens eine Testfunktion.
- Der Name einer Testfunktion greift den Namen der getesteten Funktion auf, z. B. `test_divide()` zu `divide(...)`. Bei mehreren Testfunktionen wird der getestete Aspekt ergänzt, z. B. `test_divide_by_zero_raises_error()`.
- Tests zu einer Production-Klasse werden im Testmodul in einer Testklasse `Test<Klassenname>` gebündelt. Die Testmethoden orientieren sich an der öffentlichen API der Production-Klasse.
- Freie Production-Funktionen werden durch freie Testfunktionen getestet; für sie wird keine Testklasse eingeführt.
- Die Reihenfolge der Testklassen und freien Testfunktionen folgt möglichst der Reihenfolge der getesteten öffentlichen Objekte im Production-Modul.
- Der Modulname soll nicht im Namen der Testfunktion wiederholt werden. Der Klassenname soll nicht in der Testmethode wiederholt werden, wenn die Testklasse ihn schon trägt.

Die Symmetrie bedeutet nicht, private Hilfsmethoden oder triviale Getter zu spiegeln.

## Integrationstests

- Sie liegen getrennt unter `tests/integration`.
- Die 1:1-Zuordnung aus dem Abschnitt über Unit-Tests gilt für sie nicht.

## KISS-Prinzip

> Halte auch den Test-Code einfach.

- Auch für Tests gilt das *KISS-Prinzip*. Test-Code soll möglichst klein, zielgerichtet und mit geringer mentaler Last bleiben.
- Auch bei Test-Code aktiv prüfen, ob eine kleine Vereinfachung oder ein kleines Refactoring bestehenden Test-Code vereinfachen und *LOC sparen* kann.
- Vor größerem Refactoring am Test-Code kurz beschreiben, was vereinfacht werden soll, und ein Go einholen.

## DRY-Prinzip

> Vermeide duplizierten Test-Code.

- Das *DRY-Prinzip* gilt im gesamten Workspace ausdrücklich auch für Test-Code, wenn dadurch LOC und Redundanz kleiner bleiben.
- Unnötige [Duplikation](http://xunitpatterns.com/Test%20Code%20Duplication.html) ist in Test-Code ebenso zu vermeiden wie in Production-Code.
- DRY zielt in Tests vor allem auf Wissensduplikation, nicht auf jede kleine lokale Wiederholung. Tests sollen lokal lesbar bleiben.
- Wenn mehrere Tests dasselbe wiederholen (z. B. einen Pfad oder Config-Wert), besser eine *Single Source of Truth* verwenden:
	- Benannte Konstante, [Test-Helfer](http://xunitpatterns.com/Test%20Helper.html), Fixture 
	- Parametrisierter Test: `pytest.mark.parametrize`

## Was getestet wird

### Verhalten vor Implementierung

> Tests sollen die öffentliche API priorisieren.

- Denn Tests von Interna machen Umstrukturierungen und Refactoring oft schwerer.
- Was mit öffentlicher API gemeint ist:
	- Namen mit führendem `_` gelten als private Implementierungsdetails und sollen im Regelfall nicht direkt getestet werden. Öffentlich ist ein Name grundsätzlich dann, wenn er nicht mit `_` beginnt.
	- Wenn die umgebende Struktur privat ist, z. B. ein Modul oder eine Klasse mit führendem `_`, gelten auch ihre enthaltenen Namen als privat.
	- Eine Package- oder Bibliotheks-API kann enger sein als die Menge aller öffentlichen Modulnamen. Sie wird über die dokumentierte Package-Oberfläche festgelegt, z. B. über `__init__.py` oder ein API-Modul.
- Tests sollen im Regelfall ein von außen beobachtbares Verhalten der getesteten API prüfen, nicht den internen Ablauf der Implementierung nachbauen.
- Ein Test soll möglichst aus dem Blickwinkel eines Nutzers der getesteten API formuliert sein: Gegeben ist ein Zustand oder Input, ausgeführt wird eine Operation, erwartet wird ein Ergebnis oder Effekt.

### Kernkomponenten

> Kernkomponenten sind Schwerpunkte der Tests. 

- *Kernkomponenten* sind die Komponenten, wo die *Dichte der Fachlogik* hoch ist, also wo in der Implementierung "die Musik spielt".
- Eine Kernkomponente wird als zumindest in ihrem Paket öffentlich vorausgesetzt. Eine Kernkomponente mit führendem `_` ist im Regelfall ein Design-Smell.
- Eine Kernkomponente soll im Docstring als solche erkennbar sein.

Beispiel:
```
class StudyRunner:
    """Core component for running one HPO study in a study series."""
```

## Was nicht direkt getestet wird

- Private Hilfsmethoden werden im Regelfall nicht direkt getestet.
- Triviale public Funktionen ohne nennenswerte Logik, z. B. einfache Datenzugriffe und reine Weiterleitungen, brauchen keine eigenen Tests.
- Solche Tests erhöhen oft die Kopplung an die Implementierung und machen Refactorings fragil.

Relevantes Verhalten wird stattdessen über die öffentliche API oder über Kernkomponenten getestet.

## Testdichte

Also: Die Testdichte folgt grob dieser Priorität:

> ( private Unit < triviale public Unit < ) normale public Unit < Kernkomponente

- Das Eingeklammerte wird im Regelfall nicht direkt getestet.
- Die Testdichte entspricht der Dichte der Fachlogik im public Production-Code.

## Fragile Tests vermeiden

- Wenn eine kleine produktive Umbenennung oder lokale interne Änderung viele Testanpassungen auslöst, ist das als [Fragile Test](http://xunitpatterns.com/Fragile%20Test.html)-Smell zu behandeln.
- Vor weiterem Ausbau des Test-Codes ist dann kurz zu prüfen, ob *Extract Helper*, kleine *Fixtures* oder benannte Konstanten die Duplikation verringern und die Kopplung an Implementierungsdetails reduzieren.
- Wenn ein Refactoring ohne Verhaltensänderung viele Tests bricht, ist das ein Hinweis, dass die Tests zu stark an Implementierungsdetails gekoppelt sind.

[^1]: Vgl. [Testcase Class per Class](http://xunitpatterns.com/Testcase%20Class%20per%20Class.html). Das ist das Analogon in Java, wo jede Funktion zu einer Klasse gehört.