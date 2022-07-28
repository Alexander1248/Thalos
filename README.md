FastNNLib
======
FastNNLib is an open source Java neural network framework.
It contains open source Java library which correspond to basic NN concepts. 
It has been released as open source under the Apache 2.0 license.
Adding Maven Dependency
======

Copy/Paste following code into your pom.xml file

```xml
<repositories>
    <repository>
        <id>fastnnlib-mvn-repo</id>
        <url>https://raw.github.com/Alexander1248/FastNNLib/mvn-repo/</url>
        <snapshots>
            <enabled>true</enabled>
            <updatePolicy>always</updatePolicy>
        </snapshots>
    </repository>
</repositories>
    
<dependencies>
        <dependency>
            <groupId>ru.alexander1248</groupId>
            <artifactId>fastnnlib</artifactId>
            <version>1.0.0</version>
        </dependency>
</dependencies>
```
Getting and Building from Sources using command line
======

git clone https://github.com/Alexander1248/FastNNLib.git

cd FastNNLib

mvn
