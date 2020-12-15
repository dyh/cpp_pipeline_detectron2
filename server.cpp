#include "server.h"

std::mutex mutex_ticks_map;

static Workspace workSpace_;
static caffe2::NetDef initNet_, predictNet_;

/*!
 * @brief init server
 * @param[in] host
 * @param[in] port
 * @param[in] timeout in seconds
*/
Server::Server(const string& host, int port, int timeout) {

    // model file
    string predictNetPath = "/home/dyh/workspace/unbox/unbox_cpp_caffe2/python_project/output/model.pb";
    // weights file
    string initNetPath = "/home/dyh/workspace/unbox/unbox_cpp_caffe2/python_project/output/model_init.pb";

    // initialize net and workspace
    CAFFE_ENFORCE(ReadProtoFromFile(initNetPath, &initNet_));
    CAFFE_ENFORCE(ReadProtoFromFile(predictNetPath, &predictNet_));

    for (auto& str : predictNet_.external_input()) {
        workSpace_.CreateBlob(str);
    }

    CAFFE_ENFORCE(workSpace_.CreateNet(predictNet_));
    CAFFE_ENFORCE(workSpace_.RunNetOnce(initNet_));

    // ---------

    this->host_ = host.c_str();
    this->port_ = port;
    this->timeout_seconds_ = timeout;

    // init colors_ of annotation
    RNG rng1;
    this->colors_.emplace_back(rng1.uniform(0, 255), rng1.uniform(0, 255), rng1.uniform(0, 255), 255.0);
    rng1.next();
    this->colors_.emplace_back(rng1.uniform(0, 255), rng1.uniform(0, 255), rng1.uniform(0, 255), 255.0);

    // init classes_ label of annotation
    this->classes_.emplace_back("fissure");
    this->classes_.emplace_back("water");

    // create the timeout daemon thread
    this->thread_timeout_daemon_ = thread(&Server::TimeoutHandle, this);
}

/*!
 * @brief start server
*/
[[noreturn]] void Server::Start() {


    int socket_fd = socket(AF_INET, SOCK_STREAM, IPPROTO_IP);

    if (socket_fd == -1) {
        perror("Error: socket");
    }

    auto opt = 1;
    auto error_code = setsockopt(socket_fd, SOL_SOCKET, SO_REUSEADDR, &opt, sizeof(opt));
    if (error_code == -1) {
        perror("Error: setsockopt");
    }

    // bind
    struct sockaddr_in server_addr{};
    server_addr.sin_family = AF_INET;
    server_addr.sin_port = htons(this->port_);
    server_addr.sin_addr.s_addr = inet_addr(this->host_);

    if (bind(socket_fd, (struct sockaddr *)& server_addr, sizeof(server_addr)) == -1) {
        perror("Error: bind");
    }

    // listen
    if (listen(socket_fd, 5) == -1) {
        perror("Error: listen");
    }

    while (true) {

        cout << "listening..." << endl;

        // accept
        struct sockaddr_in client_addr{};
        socklen_t client_addr_len = sizeof(client_addr);

        int new_connection_fd = accept(socket_fd, (struct sockaddr *)& client_addr, &client_addr_len);

        if (new_connection_fd < 0) {
            perror("Error: accept");
            continue;
        } else {
            char clientIP[INET_ADDRSTRLEN] = "";
            inet_ntop(AF_INET, &client_addr.sin_addr, clientIP, INET_ADDRSTRLEN);
            cout << "remote client from " << clientIP << ":" << ntohs(client_addr.sin_port) << endl;

            // generate a thread name
            long thread_name = std::clock();

            // add to timeout vector
            {
                std::lock_guard<std::mutex> lockGuard(mutex_ticks_map);
                this->map_latest_message_timestamp_[thread_name] = Server::GetCurrentTimestamp();
            }

            // 启动线程
            this->vector_threads_.emplace_back(
                    thread(&Server::SocketHandle, this, new_connection_fd, clientIP, thread_name));
        }

        // sleep 1 micro_seconds
        usleep(1);
    }

    shutdown(socket_fd, SHUT_RDWR);

    int thread_count = this->vector_threads_.size();

    for (int i = 0; i < thread_count; i++) {
        this->vector_threads_.at(i).join();
    }

    this->thread_timeout_daemon_.join();

    // release vector
    this->vector_threads_.clear();
    vector<thread>().swap(this->vector_threads_);

    // release map
    this->map_latest_message_timestamp_.clear();
    map<long, long>().swap(this->map_latest_message_timestamp_);

}


/*!
 * @brief to judge whether the socket has timeout
*/
void Server::TimeoutHandle() {

    auto timeout_value_ms = this->timeout_seconds_ * 1000;

    while (true) {

        auto timestamp_now = Server::GetCurrentTimestamp();

        long output_thread_name = -1;

        {
            std::lock_guard<std::mutex> lockGuard(mutex_ticks_map);

            if (find_if(this->map_latest_message_timestamp_.begin(),
                        this->map_latest_message_timestamp_.end(),

                        [timestamp_now, timeout_value_ms, &output_thread_name]
                        (map<long, long>::value_type& item) {
                            auto ret = ((timestamp_now - item.second) > timeout_value_ms);
                            if (ret) {
                                output_thread_name = item.first;
                            }
                            return ret;
                        })
                        != this->map_latest_message_timestamp_.end()) {

                cout << "# thread [" << output_thread_name << "] is timeout, removed from map_timestamp" << endl;
                this->map_latest_message_timestamp_.erase(output_thread_name);
            } else {
                // cout << "do not timeout: " << output_thread_name << endl;
            }
            // cout << "timestamp count: " << map_latest_message_timestamp_.size() << endl;
        }

//        cout << ".." << endl;

        // sleep in seconds
        sleep(this->timeout_seconds_);
    }

}

/*!
 * @brief function of handle socket
 * @param[in] connection_fd
 * @param[in] client_address
 * @param[in] thread_name
*/
void Server::SocketHandle(int connection_fd, const string & client_address, long thread_name) {

    cout << "accepted connection from " << client_address << endl;
    auto message = ::Message(connection_fd, client_address);

    // receive messages from socket client
    // flag of loop
    while (true) {
        // if this thread is not timeout
        {
            std::lock_guard<std::mutex> lockGuard(mutex_ticks_map);

            if (find_if(this->map_latest_message_timestamp_.begin(),
                        this->map_latest_message_timestamp_.end(),
                        [thread_name]
                                (map<long, long>::value_type& item) { return item.first == thread_name; })
                != this->map_latest_message_timestamp_.end()) {

            } else {
                cout << "do not find [" << thread_name << "] in map_loop, BREAK while loop" << endl;
                break;
            }
        }

        //clear buffer and response flag
        message.Clear();

        unsigned char *output_buffer;
        long output_length = 0;

        // if the socket works well
        if (message.Read()) {

            message.GetImageBufferResult(output_buffer, output_length);

            if (output_length > 0) {

                std::vector<uchar> vector_image(output_buffer, output_buffer + output_length);

                cout << "# received " << output_length << " bytes from client " << client_address << ", in thread [" << thread_name << "]"  <<  endl;

                cv::Mat mat_input;
                cv::imdecode(vector_image, cv::ImreadModes::IMREAD_COLOR, &mat_input);

                // release vector
                vector_image.clear();
                vector<uchar>().swap(vector_image);

                // TODO: begin, process cv::Mat image object here

                const int height = mat_input.rows;
                const int width = mat_input.cols;
                // FPN models require divisibility of 32
                assert(height % 32 == 0 && width % 32 == 0);

                // setup inputs
                auto data = BlobGetMutableTensor(workSpace_.GetBlob("data"), caffe2::CPU);
                data->Resize(this->batch_, this->channels_, height, width);
                auto *ptr = data->mutable_data<float>();
                // HWC to CHW
                for (int c = 0; c < 3; ++c) {
                    for (int i = 0; i < height * width; ++i) {
                        ptr[c * height * width + i] = static_cast<float>(mat_input.data[3 * i + c]);
                    }
                }

                auto im_info = BlobGetMutableTensor(workSpace_.GetBlob("im_info"), caffe2::CPU);
                im_info->Resize(this->batch_, this->channels_);
                auto *im_info_ptr = im_info->mutable_data<float>();
                im_info_ptr[0] = height;
                im_info_ptr[1] = width;
                im_info_ptr[2] = 1.0;

                // run the network
                CAFFE_ENFORCE(workSpace_.RunNet(predictNet_.name()));

//                // run 3 more times to benchmark
//                //        int N_benchmark = 3;
//                int N_benchmark = 1;
//
//                auto start_time = chrono::high_resolution_clock::now();
//                for (int i = 0; i < N_benchmark; ++i) {
//                    CAFFE_ENFORCE(workSpace_.RunNet(predictNet_.name()));
//                }
//                auto end_time = chrono::high_resolution_clock::now();
//                auto ms = chrono::duration_cast<chrono::microseconds>(end_time - start_time)
//                        .count();
//                cout << "Latency (should vary with different inputs): "
//                     << ms * 1.0 / 1e6 / N_benchmark << " seconds" << endl;

                // parse Mask R-CNN outputs
                caffe2::Tensor bbox(
                        workSpace_.GetBlob("bbox_nms")->Get<caffe2::Tensor>(), caffe2::CPU);
                caffe2::Tensor scores(
                        workSpace_.GetBlob("score_nms")->Get<caffe2::Tensor>(), caffe2::CPU);
                caffe2::Tensor labels(
                        workSpace_.GetBlob("class_nms")->Get<caffe2::Tensor>(), caffe2::CPU);
                caffe2::Tensor mask_probs(
                        workSpace_.GetBlob("mask_fcn_probs")->Get<caffe2::Tensor>(), caffe2::CPU);

                cout << "bbox:" << bbox.DebugString() << endl;
                cout << "scores:" << scores.DebugString() << endl;
                cout << "labels:" << labels.DebugString() << endl;
                cout << "mask_probs: " << mask_probs.DebugString() << endl;

                int num_instances = bbox.sizes()[0];
                for (int i = 0; i < num_instances; ++i) {
                    float score = scores.data<float>()[i];
                    if (score < 0.6)
                        continue; // skip them

                    const float *box = bbox.data<float>() + i * 4;
                    int label = labels.data<float>()[i];

                    cout << "Prediction " << i << ", xyxy=(";
                    cout << box[0] << ", " << box[1] << ", " << box[2] << ", " << box[3]
                         << "); score=" << score << "; label=" << label << endl;

                    auto rect_box = Rect(box[0], box[1], box[2] - box[0], box[3] - box[1]);

                    const float *mask = mask_probs.data<float>() +
                                        i * mask_probs.size_from_dim(1) + label * mask_probs.size_from_dim(2);

                    // save the 28x28 mask
                    Mat mat_mask(28, 28, CV_32F);
                    memcpy(mat_mask.data, mask, 28 * 28 * sizeof(float));
                    mat_mask = mat_mask * 255.;

                    // draw box, mask and label text on the origin image
                    this->DrawBox(mat_input, label, score, rect_box, mat_mask);

                    // release the 28x28 mask object
                    mat_mask.release();
                }

                // TODO: done.

                // send cv::Mat image to client
                bool is_write_succeed = message.WriteImage(mat_input);

                // release cv::Mat image object
                    mat_input.release();

                if (!is_write_succeed)
                {
                    // meet socket error
                    cout << "write socket error, remote socket maybe closed, BREAK while loop" << endl;
                    break;
                }

            }

        } else {
            // if we got error of socket, break while loop
            cout << "read socket error, remote socket maybe closed, BREAK while loop" << endl;

            break;
        }

        // judge whether the thread is not timeout

        {
            std::lock_guard<std::mutex> lockGuard(mutex_ticks_map);

            if (find_if(this->map_latest_message_timestamp_.begin(),
                        this->map_latest_message_timestamp_.end(),
                        [thread_name]
                                (map<long, long>::value_type& item) { return item.first == thread_name; })
                != this->map_latest_message_timestamp_.end()) {

                this->map_latest_message_timestamp_[thread_name] = Server::GetCurrentTimestamp();

            } else {
                cout << "do not find [" << thread_name << "] in map_loop, BREAK while loop" << endl;
                break;
            }
        }

        // sleep
        usleep(1);
    }

    cout << "shutdown connection_fd" << endl;
    shutdown(connection_fd, SHUT_RDWR);
}


/*!
 * @brief get current time ticks
 * @return ticks in ms long
*/
long Server::GetCurrentTimestamp() {
    std::chrono::time_point<std::chrono::system_clock, std::chrono::milliseconds> tp = std::chrono::time_point_cast<std::chrono::milliseconds>(
            std::chrono::system_clock::now());
    return tp.time_since_epoch().count();
}


/*!
 * @brief draw box, mask and label text on the origin image
 * @param[out] frame origin image
 * @param[in] classId annotation classes_ id
 * @param[in] conf confidence
 * @param[in] box bbox
 * @param[in] objectMask Mask of objects
*/
void Server::DrawBox(Mat& frame, int classId, float conf, Rect box, Mat& objectMask)
{
    // get color by different classId
    auto color1 = this->colors_[classId];

    //draw a rectangle to display the bounding box
    rectangle(frame, Point(box.x, box.y), Point(box.x + box.width, box.y + box.height), color1, 1);

    //generate a label text for the class name and its confidence
    string label = to_string(conf);
    if (!this->classes_.empty())
    {
        CV_Assert(classId < (int)this->classes_.size());
        // get class name, fissure or water
        label = this->classes_[classId] + ":" + label;
    }

    //display the label at the top of the bounding box
    int baseLine;
    Size labelSize = getTextSize(label, FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseLine);
    box.y = max(box.y, labelSize.height);
//    rectangle(frame, Point(box.x, box.y - round(1.5 * labelSize.height)), Point(box.x + round(1.5 * labelSize.width), box.y + baseLine), color1, FILLED);
    // draw a filled rectangle around the text
    rectangle(frame, Point(box.x, box.y - round(labelSize.height)), Point(box.x + round(labelSize.width), box.y + baseLine), color1, FILLED);
    // draw annotation text
    putText(frame, label, Point(box.x, box.y), FONT_HERSHEY_SIMPLEX, 0.50, cv::Scalar(255, 255, 255), 1);

    cv::Size dSize = cv::Size(box.width, box.height);

    // resize mask matrix from w:28 h:28 to w:800 h:800
    Mat mat_temp(dSize, CV_32F);
    resize(objectMask, mat_temp, dSize, INTER_LINEAR);
    Mat mat_coloredRoi = (0.3 * color1 + 0.7 * frame(box));
    mat_coloredRoi.convertTo(mat_coloredRoi, CV_8U);

    // draw the contours on the image
    vector<Mat> contours;
    Mat hierarchy;
    mat_temp.convertTo(mat_temp, CV_8U);
    findContours(mat_temp, contours, hierarchy, RETR_CCOMP, CHAIN_APPROX_SIMPLE);
    drawContours(mat_coloredRoi, contours, -1, color1, 2, LINE_8, hierarchy, 100);
    mat_coloredRoi.copyTo(frame(box), mat_temp);

    contours.clear();
    vector<Mat>().swap(contours);

    mat_temp.release();
    mat_coloredRoi.release();
}
