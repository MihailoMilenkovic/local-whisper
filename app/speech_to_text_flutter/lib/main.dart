import 'dart:async';
import 'dart:convert';

import 'package:flutter/material.dart';
import 'package:http/http.dart' as http;

import 'config.dart';

void main() {
  runApp(const MyApp());
}

class MyApp extends StatelessWidget {
  const MyApp({super.key});

  @override
  Widget build(BuildContext context) {
    return const MaterialApp(
      home: MyHomePage(),
    );
  }
}

class MyHomePage extends StatefulWidget {
  const MyHomePage({super.key});

  @override
  State<MyHomePage> createState() => _MyHomePageState();
}

class _MyHomePageState extends State<MyHomePage> {
  late http.Client client;
  late http.Response response;
  late StreamController<String> wordStreamController;

  @override
  void initState() {
    super.initState();
    client = http.Client();
    wordStreamController = StreamController<String>();

    // Replace the URL with your API endpoint that provides streaming data
    http
        .get(Uri.parse('${AppConfig.whisperURL}/${AppConfig.whisperEndpoint}'))
        .then((http.Response response) {
      if (response.statusCode == 200) {
        // Handle the streaming response here
        StreamController<List<int>> streamController =
            StreamController<List<int>>();
        streamController.add(response.bodyBytes);

        streamController.stream
            .transform(utf8.decoder)
            .transform(const LineSplitter())
            .listen((String line) {
          // Split the line into words and send them to the stream controller
          List<String> words = line.split(' ');
          for (var word in words) {
            wordStreamController.add(word);
          }
        });
      } else {
        throw Exception('Failed to load data');
      }
    });
  }

  @override
  void dispose() {
    client.close();
    wordStreamController.close();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: const Text('Speech to text app'),
      ),
      body: Center(
        child: StreamBuilder<String>(
          stream: wordStreamController.stream,
          builder: (context, snapshot) {
            if (snapshot.hasData) {
              return Text(
                snapshot.data ?? '', //display empty snapshot as ''
                style: const TextStyle(fontSize: 20),
              );
            } else if (snapshot.hasError) {
              return Text('Error: ${snapshot.error}');
            } else {
              return const CircularProgressIndicator();
            }
          },
        ),
      ),
    );
  }
}
