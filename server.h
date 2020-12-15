#ifndef SERVER_SERVER_H
#define SERVER_SERVER_H

#include <ctime>
#include <iostream>
#include <thread>
#include <vector>
#include <sys/socket.h>
#include <netinet/in.h>
#include <unistd.h>
#include <arpa/inet.h>
#include <mutex>
#include <cassert>
#include <chrono>
#include <dirent.h>
#include <c10/util/Flags.h>

#include <caffe2/core/blob.h>
#include <caffe2/core/init.h>
#include <caffe2/core/workspace.h>
#include <caffe2/utils/proto_utils.h>
#include <opencv2/opencv.hpp>

#include "json.hpp"
#include "message.h"


using namespace std;
using namespace cv;
using namespace std::chrono;
using namespace caffe2;

class Server {

public:
    Server(const string& host, int port, int timeout);

    [[noreturn]] void Start();

private:

    // host address
    const char *host_;

    //port number
    int port_;

    // seconds of timeout
    int timeout_seconds_;

    // vector of socket threads
    vector<thread> vector_threads_;

    // record the flag of loop and latest std::clock of each socket message, <thread_name, ticks_value>
    map<long, long> map_latest_message_timestamp_;

    vector<string> classes_;
    vector<cv::Scalar> colors_;

    const int batch_ = 1;
    const int channels_ = 3;

    // daemon thread of timeout
    thread thread_timeout_daemon_;

    // socket function in thread
    void SocketHandle(int connection_fd, const string& client_address, long thread_name);

    // timeout function in thread
    [[noreturn]] void TimeoutHandle();

    // get current time ticks
    static long GetCurrentTimestamp();

    void DrawBox(Mat& frame, int classId, float conf, Rect box, Mat & objectMask);



};



#endif //SERVER_SERVER_H
