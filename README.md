FastNNLib
======
FastNNLib is an open source Java neural network framework.
It contains open source Java library which correspond to basic NN concepts. The library is also optimized for fast calculations on GPU and CPU. This makes it possible to calculate and train neural networks at high speeds with minimal losses in development efficiency.
It has been released as open source under the Apache 2.0 license.

Adding Maven Dependency
======

Copy/Paste following code into your pom.xml file

```xml
<repositories>
    <repository>
        <id>fastnnlib</id>
        <url>https://raw.github.com/Alexander1248/FastNNLib/mvn-repo/</url>
    </repository>
</repositories>
    
<dependencies>
        <dependency>
            <groupId>ru.alexander1248</groupId>
            <artifactId>fastnnlib</artifactId>
            <version>1.1.5</version>
        </dependency>
</dependencies>
```

Getting and Building from Sources using NetBeans
======

Click: Main Menu > Team > Git > Clone

For Repository URL enter https://github.com/Alexander1248/FastNNLib.git

Click Finish

Right click cloned project, and click Build

Getting and Building from Sources using command line
======

git clone https://github.com/Alexander1248/FastNNLib.git

cd FastNNLib

mvn
