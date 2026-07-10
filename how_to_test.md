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

- Zu einem Production-Modul gibt es ein korrespondierendes Test-Modul.
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

- Auch für Tests gilt das KISS-Prinzip. Test-Code soll möglichst klein, zielgerichtet und mit geringer mentaler Last bleiben.
- Auch bei Test-Code aktiv prüfen, ob eine kleine Vereinfachung oder ein kleines Refactoring bestehenden Test-Code vereinfachen und `LOC` sparen kann.
- Vor größeren Refactorings am Test-Code kurz beschreiben, was vereinfacht werden soll, und ein Go einholen.

## DRY-Prinzip

> Vermeide duplizierten Test-Code.

- Das DRY-Prinzip gilt im gesamten Workspace ausdrücklich auch für Test-Code, wenn dadurch `LOC` und Redundanz kleiner bleiben.
- Unnötige Duplikation ist in Test-Code ebenso zu vermeiden wie in Production-Code.
- DRY zielt in Tests vor allem auf Wissensduplikation, nicht auf jede kleine lokale Wiederholung. Tests sollen lokal lesbar bleiben.
- Wenn mehrere Tests dasselbe wiederholen (z. B. einen Pfad oder Config-Wert), ist eine kleine gemeinsame Test-Hilfe, Konstante oder Fixture zu bevorzugen (`Single Source of Truth`).

## Was getestet wird

### Verhalten vor Implementierung

> Tests sollen die öffentliche API priorisieren.

- Denn Tests von Interna machen Umstrukturierungen und Refactorings oft schwerer.
- Die öffentliche API ist, was über das zuständige `__init__.py` öffentlich gemacht wird. Das betrifft Module, Klassen und Funktionen.  
- Tests sollen im Regelfall ein von außen beobachtbares Verhalten der getesteten API prüfen, nicht den internen Ablauf der Implementierung nachbauen.
- Ein Test soll möglichst aus dem Blickwinkel eines Nutzers der getesteten API formuliert sein: Gegeben ist ein Zustand oder Input, ausgeführt wird eine Operation, erwartet wird ein Ergebnis oder Effekt.

### Kernkomponenten

> Kernkomponenten sind Schwerpunkte der Tests. 

- Das gilt auch für Kernkomponenten, die nicht direkt durch die öffentliche API aufgerufen werden (quasi "innere Organe").
- Kernkomponenten sind die Komponenten, wo die Dichte der Fachlogik hoch ist, wo "die Musik spielt".
## Was nicht direkt getestet wird

- Private Hilfsmethoden werden im Regelfall nicht direkt getestet.
- Triviale Getter, einfache Datenzugriffe und reine Weiterleitungen brauchen keine eigenen Tests.
- Solche Tests erhöhen oft die Kopplung an die Implementierung und machen Refactorings fragil.

Relevantes Verhalten wird stattdessen über die öffentliche API oder über Kernkomponenten getestet.

## Fragile Tests vermeiden

- Wenn eine kleine produktive Umbenennung oder lokale interne Änderung viele Testanpassungen auslöst, ist das als `Fragile Test`-Smell zu behandeln.
- Vor weiterem Ausbau des Test-Codes ist dann kurz zu prüfen, ob `Extract Helper`, kleine Fixtures oder benannte Konstanten die Duplikation verringern und die Kopplung an Implementierungsdetails reduzieren.
- Wenn ein Refactoring ohne Verhaltensänderung viele Tests bricht, ist das ein Hinweis, dass die Tests zu stark an Implementierungsdetails gekoppelt sind.
