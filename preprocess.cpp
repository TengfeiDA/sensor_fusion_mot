#include<iostream>
#include <fstream>
#include <string>
#include "json/json.h"

using namespace std;

void readJsonFromStr()
{
	const char* str = "{\"name\":\"weier\",\"age\":21,\"sex\":\"man\"}";
	Json::Reader reader;
	Json::Value root;

	if (reader.parse(str, root))
	{
		string name = root["name"].asString();
		int age = root["age"].asInt();
		string sex = root["sex"].asString();
		cout << '{' << endl;
		cout << '\t' << "\"name\"" << '\t' << ':' << '\t' << name << ',' << endl;
		cout << '\t' << "\"age\"" << '\t' << ':' << '\t' << age << ',' << endl;
		cout << '\t' << "\"sex\"" << '\t' << ':' << '\t' << sex << endl << '}' << endl;
	}
}

void readDataFromJsonFile()
{
	Json::Reader reader;/*用于按照JSON数据格式进行解析*/
	Json::Value root;/*用于保存JSON类型的一段数据*/
	
	ifstream srcFile("test.json", ios::binary);/*定义一个ifstream流对象，与文件demo.json进行关联*/
	if (!srcFile.is_open())
	{
		cout << "Fail to open src.json" << endl;
		return;
	}
	/*将demo.json数据解析到根节点root*/
	if (reader.parse(srcFile, root))
	{
		/*读取根节点信息*/
		string myName = root["name"].asString();
		string mySex = root["sex"].asString();
		int myAge = root["age"].asInt();
		cout << "My Info:" << endl;
		cout << "name : " << myName << endl;
		cout << "sex : " << mySex << endl;
		cout << "age : " << myAge << endl;
		/*读取子节点信息*/
		cout << "My friends:" << endl;
		for (int i = 0; i < root["friend"].size(); i++)
		{
			cout << "name : " << root["friend"][i]["name"].asString() << endl;
		}
	}
	srcFile.close();
}


int main(int argc, char** argv) {

  cout << "It's preprocess.\n";

  readDataFromJsonFile();

  return 0;
}