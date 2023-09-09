This library is currently outdated.
======
A new library is currently being developed. 
Thalos
======
Thalos is an open source Java neural network framework.
It contains open source Java library which correspond to basic NN concepts. The library is also optimized for fast calculations on GPU and CPU. This makes it possible to calculate and train neural networks at high speeds with minimal losses in development efficiency.
Product powered by Aparapi.
It has been released as open source under the Apache 2.0 license.

### Notification
Since version 2.0.0, the library has been completely rewritten. As a result, there is a big difference in functions and classes between the latest version before 2.0.0 and after it.

Adding Maven Dependency
======

Copy/Paste following code into your pom.xml file.

Up to version 1.5.0:

```xml
<repositories>
    <repository>
        <id>ru.alexander1248</id>
        <url>https://github.com/Alexander1248</url>
    </repository>
</repositories>
    
<dependencies>
        <dependency>
            <groupId>ru.alexander1248</groupId>
            <artifactId>fastnnlib</artifactId>
            <version>1.3.5</version>
        </dependency>
</dependencies>
```

After version 1.5.0:

```xml
<repositories>
    <repository>
        <id>ru.alexander1248</id>
        <url>https://github.com/Alexander1248</url>
    </repository>
</repositories>
    
<dependencies>
        <dependency>
            <groupId>ru.alexander1248</groupId>
            <artifactId>thalos</artifactId>
            <version>1.5.0</version>
        </dependency>
</dependencies>
```

Getting and Building from Sources using NetBeans
======

Click: Main Menu > Team > Git > Clone

For Repository URL enter https://github.com/Alexander1248/Thalos.git

Click Finish

Right click cloned project, and click Build

Getting and Building from Sources using command line
======

git clone https://github.com/Alexander1248/Thalos.git

cd Thalos

mvn
