From Tale to Speech: Ontology-based Emotion and
Dialogue Annotation of Fairy Tales with a TTS Output

Christian Eisenreich1, Jana Ott1, Tonio Süßdorf1, Christian Willms1,

Thierry Declerck2,1

1 Saarland  University,  Computational  Linguistics  Department,  D-66041 Saarbrücken,  Ger-

many

(eisenr|janao|tonios|cwillms)@coli.uni-saarland.de
2  German  Research  Center  for  Artificial  Intelligence  (DFKI),  Language  Technology  Lab,

Stuhlsatzenhausweg 3, D-66123 Saarbrücken, Germany

thierry.declerck@dfki.de

Abstract.  In  this demo  and  poster  paper,  we  describe the  concept  and  imple-
mentation of an ontology-based storyteller for fairy tales. Its main functions are
(i)  annotating  the  tales  by  extracting timeline  information,  characters  and  dia-
logues with corresponding emotions expressed in the utterances, (ii) populating
an  existing  ontology  for  fairy  tales  with  the  previously  extracted  information
and (iii) using this ontology to generate a spoken version of the tales.

Common  natural  language  processing  technologies  and  resources,  such  as
part-of-speech tagging, chunking and semantic networks have been successfully
used for the implementation of the three tasks mentioned just above, including
the integration of an open source text-to-speech system. The code of the system
is publicly available.

Keywords: ontology, natural language processing, text-to-speech, semantic-
network, fairy tale, storytelling

1

Introduction

The idea of developing an ontology-based storyteller for fairy tales was based on the
consideration of two previous works in the field of narrative text processing. The first
work  is  described  in  (Scheidel  &  Declerck,  2010),  which  is  about  an  augmented
Proppian1 fairy tale markup language, called Apftml, which we extended according to
the needs of our current work.

Our second starting point is described in (Declerck et al., 2012), which presents an
ontology-based  system  that  is  able  to  detect  and  recognize  the  characters  (partici-
pants)  playing  a  role  in  a  folktale.  Our  system  combines  and  extends  the  results  of

1   From   „Vladimir  Yakovlevich  Propp”,   who  was  “a  Soviet  folklorist  and  scholar  who
analyzed the basic plot components of Russian folk tales to identify their simplest
irreducible narrative elements.”  (http://en.wikipedia.org/wiki/Vladimir_Propp)

adfa, p. 1, 2011.
© Springer-Verlag Berlin Heidelberg 2011

those studies, adding the detection of dialogues and emotions in the tales and an on-
tology-driven Text-To-Speech (TTS) component  that  “reads”  the  tales,  with  individu-
al  voices  for  every  character, including  also  a  voice  for the narrator,  and  taking into
account the types of emotions detected during the textual processing of the tales.

To  summarize:  Our  system  first parses  the input tale  (in  English  or  German)  and
extracts as much relevant information as possible  on the characters – including their
emotions  --  and the  events  they  are involved  in. This  provides  us  with an annotated
version of the tale that is used for populating the ontology. The system finally uses the
ontology and a robust and parameterizable TTS system to generate the speech output.
All  the  data  of  the  system  have  been  made  available  in  a  bitbucket  repository
(https://bitbucket.org/ceisen/apftml2repo),  including  documentation  and  related  in-
formation2.

2

Architecture of the System

Firstly, we use the Python NLTK3 and Pattern API4 to annotate the tale. Then we use
the Java OWL-API5 to populate the ontology. And finally  the Mary Text-To-Speech
system6 is  used  to  generate  the  speech  output.  Mary  is  an  open-source,  multilingual
Text-to-Speech  Synthesis platform,  which  is robust,  easy  to  configure and allows  us
to extend our storyteller to more languages. The general architecture of the system is
displayed below in Fig. 1.

Fig. 1. The general architecture of the ontology-driven ‘Tale to Speech” system

2   An  example  of  the  audio  data  generated  for  the  tale  “The  Frog  Prince”  is  available  at
https://bytebucket.org/ceisen/apftml2repo/raw/763c5eb533f09997e757ec61652310c742238
384/example%20output/audio_output.mp3.

3   Natural Language Toolkit: http://www.nltk.org/. See also (Bird et al., 2009)
4   See (De Smedt & Daelemans, 2012).
5   See (Horridge & Bechhofer, 2011).
6   http://mary.dfki.de/.  See also  (Schröder  Marc  &Trouvain,  2003)  or (Charfuelan  & Steiner,

2013).

3

The Ontology Population

The ontology  we use is an extension of the one presented in (Declerck et al., 2012),
which describes basically family structures among human beings, but also a small list
of extra-natural beings. In the extended version of the ontology we include also tem-
poral information (basically  for representing the mostly linear structure of the narra-
tive)  as  well  as  dialogue  structures,  including  the  participants  involved  in  the  dia-
logues (sender(s) and receivers(s)), whereas we give special attention also to the nar-
rator   of   the   tale,   since   this   “character”   is   also   giving   relevant   information  about  the
status  of  the  characters  in  the  tales,  including  their  emotional  state.  Dialogues  are
synchronized with the linear narrative structure. Detected emotions are also included
in the populated ontology, and are attached for the time being to utterances, and will
be attached in the future to the characters directly. The Mary TTS system is accessing
all  this information  in  order  to  parameterize  the  voices  that  are attached  to  each  de-
tected characters.

4

A Gold Standard

In order to support evaluation of the automated annotation of fairy tales with our inte-
grated  set  of  tools  5  fairy  tales  have  been  manually  annotated7.    The  tales  are  “The
Frog  Prince”,  “The  Town  Musicians  of  Bremen”,  “Die   Bremer   Stadt   Musikanten”
(the German original version), “The  Magic  Swan  Geese”  and  “Rumpelstiltskin”.

The annotation examples show the different steps involved in the system: the text
analysis,  the  temporal  segmentation,  the  recognition  of  the  characters  and  the  dia-
logues they are involved in, the emotions that are attached to the utterances and deliv-
ered during speech the story in near real time.

5

Summary and Outlook

We have designed and implemented in the field of fairy tales an ontology-based emo-
tion- and dialogue annotation system with speech output. The system provides robust
results  for  the  tested  fairy  tales.  While  the  annotation  and  ontology  population  pro-
cesses are working for both English and German texts, the TTS output is for the time
being optimized for the English language.

Future work can deal with adding a graphical user interface, extending the parsing
process for annotating tales in other languages and populating the ontology with more
information, like the Proppian functions.

7   The  manually  annotated  tales,  together  with  the  annotation  schema,  are  available  at
https://bitbucket.org/ceisen/apftml2repo/src/763c5eb533f09997e757ec61652310c74223838
4/soproworkspace/SoPro13Java/gold/?at=master

6

References

1.  Horridge Matthew and Bechhofer Sean (2011). The owl api: A java api for owl ontologies

 IOS Press, IOS Press volume 2 number 1, 11--12

2.  Schröder Marc and Trouvain Jürgen (2003). The German text-to-speech synthesis system
MARY: A tool for research, development and teaching. Springer: International Journal of
Speech Technology, volume 6 number 4, 365—377.

3.  Marcela  Charfuelan  and  Ingmar  Steiner  (2013).  Expressive  speech  synthesis  in  MARY

TTS using audiobook data and EmotionML. ISCA: Proceedings of Interspeech 2013

4.  Steven  Bird,  Ewan  Klein,  and  Edward  Loper  (2009).  Natural  Language  Processing  with
Python---  Analyzing  Text  with  the  Natural  Language  Toolkit..  O'Reilly  Media,
(http://www.nltk.org/book/)

5.  Ekman Paul (1999). Emotions In T. Dalgleish and T. Power (Eds.) The Handbook of Cog-

nition and Emotion Pp. 45-60. Sussex, UK: John Wiley \& Sons, Ltd.

6.  De  Smedt,  Tom  and  Daelemans,  Walter  (2012).  Pattern  for  python.  The  Journal  of  Ma-

chine Learning Research, volume{13} nr.1 2063--2067

7.  Scheidel  Antonia  and  Declerck  Thierry  (2010).  Apftml-augmented  proppian  fairy  tale
markup language. First International AMICUS Workshop on Automated Motif Discovery
in Cultural Heritage and Scientific Communication Texts.. Szeged University, volume 10
8.  Declerck  Thierry,  Koleva  Nikolina  and  Krieger  Hans-Ulrich  (2012).  Ontology-based  in-
cremental annotation of characters in folktales Association for Computational Linguistics.
Proceedings  of  the  6th  Workshop  on  Language  Technology  for  Cultural  Heritage,  Social
Sciences, and Humanities, 30--34

9.  Propp  V.Y. Morphology  of the  Folktale.  Leningrad, 1928;  English:  The  Hague:  Mouton,

1958; Austin: University of Texas Press, 1968.

10.  Inderjeet Mani: Computational Modeling of Narrative. Synthesis Lectures on Human Lan-

guage Technologies, Morgan & Claypool Publishers 2012.

