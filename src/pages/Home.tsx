import React, { useCallback, useEffect, useRef, useState } from 'react';
import { Box, VStack, Text, Button, Avatar, Separator, Flex, IconButton, Image, Heading, Textarea, Link } from "@chakra-ui/react"
import { FiPlus, FiSettings, FiHelpCircle } from "react-icons/fi"
import { GoSidebarExpand } from 'react-icons/go';
import { IoSearchSharp, IoSend } from 'react-icons/io5';
import { LuMessageSquareText } from 'react-icons/lu';

// @ts-ignore
import logo from "../assets/logo.png";
import { useNavigate, useParams } from 'react-router';
import useFetch from '@/hooks/useFetch';
import { showAlert } from '@/utils/helpers';
import { MdDelete, MdContentCopy, MdThumbUp, MdThumbDown, MdFeedback, MdThumbDownOffAlt, MdThumbUpOffAlt } from 'react-icons/md';
import { IoIosCopy } from 'react-icons/io';


type Sender = "user" | "bot";

interface ChatTitle {
    id: number;
    title: string;
    created_at: string;
}

interface SidebarProps {
    allTitles: ChatTitle[];
    onSelect: (id: number) => void;
    onNewChat?: () => void;
    handleDeleteTitle: (e: React.MouseEvent<HTMLButtonElement | HTMLDivElement>, id: number) => void;
}

interface ChatMessage {
    id?: number;
    title_id: number;
    sender: Sender;
    message: string;
    created_at?: string;
    reaction?: 0 | 1 | null;
}
interface FeedbackMap {
    [chatId: string]: 0 | 1 | null;
}

interface NewChatResponse {
    title_id: string;
}



const Home: React.FC = () => {
    const { titleId, userId } = useParams<{ titleId?: string; userId?: string }>();
    const navigate = useNavigate();

    const bottomRef = useRef<HTMLDivElement>(null);

    const { loading, request } = useFetch();

    const [value, setValue] = useState<string>("");
    const [allTitles, setAllTitles] = useState<ChatTitle[]>([]);
    const [allChats, setAllChats] = useState<ChatMessage[]>([]);
    const [isAiThinking, setIsAiThinking] = useState<boolean>(false);
    const [reportData, setReportData] = useState<string>("");
    const [feedbackMap, setFeedbackMap] = useState<FeedbackMap>({});
    const [copiedChatId, setCopiedChatId] = useState<number | null>(null);

    /** Scroll to bottom when chats update */
    useEffect(() => {
        bottomRef.current?.scrollIntoView({ behavior: "smooth" });
    }, [allChats]);

    /** Fetch chat history */

    const getSidebarData = useCallback(async (userId?: string) => {
        if (!userId) return;
        try {
            const res = await request({ url: `/chatbot/getChatTitle?user_id=${userId}` });
            if (res?.success) setAllTitles(res.data);
        } catch (err) {
            console.error("Error fetching chats:", err);
        }
    }, [request]);

    const getAllChats = useCallback(async (id?: string) => {
        if (!id) return;
        try {
            const res = await request({ url: `/chatbot/getAllChats?title_id=${id}` });
            if (res?.success) {
                setAllChats(res.data)
            };
        } catch (err) {
            console.error("Error fetching chats:", err);
        }
    }, [request]);

    /** Send a chat message (user or bot) */
    const sendResponse = useCallback(async (message: string, sender: Sender, title_id?: string) => {
        try {
            await request({
                url: `/chatbot/addChat`,
                method: "POST",
                body: { message, sender, title_id: title_id ?? titleId },
            });
        } catch (err) {
            console.error("Error sending message:", err);
        }
    }, [request, titleId]);

    /** Start a new chat session */
    const sendNewChat = useCallback(async (chat: ChatMessage): Promise<string | null> => {

        try {
            const res = await request({
                url: `/chatbot/newChat`,
                method: "POST",
                body: { user_id: userId, chats: chat },
            });

            return res?.chat_id ?? null;
        } catch (err) {
            console.error("Error creating new chat:", err);
            return null;
        }
    }, [request, userId]);

    /** Handle user sending a message */
    const handleSubmit = useCallback(async (event: React.FormEvent) => {
        event.preventDefault();
        if (!value.trim()) return;

        let isNewChat = false;
        // let firstChunk = false;

        const message: ChatMessage = { message: value, sender: "user", title_id: Number(titleId) };
        const updatedChats = [...allChats, message];
        setAllChats(updatedChats);
        setValue("");
        try {
            setIsAiThinking(true);
            let currentTitleId = titleId;

            if (!currentTitleId) {
                isNewChat = true;
                const newTitleId = await sendNewChat(message);
                if (newTitleId) {
                    currentTitleId = newTitleId;
                    navigate(`/home/${userId}/${currentTitleId}`);
                    await getSidebarData(userId);
                }
            } else {
                await sendResponse(value, "user");
            }

            // const botRes = await request({ url: `/ai/get-info?query=${encodeURIComponent(value)}`, method: "GET" });

            const query = encodeURIComponent(value);
            const url = `${import.meta.env.VITE_BACKEND_URL}/ai/get-info?query=${query}&title_id=${currentTitleId}&user_id=${userId}`;
            const eventSource = new EventSource(url);

            let collected = "";
            console.log('here', isAiThinking);

            if (!isNewChat) setAllChats((prev) => [...prev, { message: "", sender: "bot", title_id: Number(currentTitleId) }]);

            eventSource.onmessage = async (e) => {
                if (!e.data) return;

                // if (!firstChunk && e.data !== "end") {
                //     firstChunk = true;
                //     setIsAiThinking(false);
                // }

                if (e.data === "end") {
                    console.log(collected);

                    setAllChats((prev) => {
                        const updated = [...prev];
                        const botIndex = updated.findIndex(c => c.sender === "bot" && c.message === "");
                        if (botIndex !== -1) updated[botIndex].message = collected;
                        return updated;
                    });


                    await sendResponse(collected, "bot", currentTitleId);
                    await getAllChats(currentTitleId);
                    setIsAiThinking(false);
                    eventSource.close();
                    return;
                }

                collected += e.data;
                if (!isNewChat) {
                    setAllChats((prev) => {
                        const updated = [...prev];
                        const lastBotIndex = updated.map(c => c.sender).lastIndexOf("bot");
                        if (lastBotIndex !== -1) {
                            updated[lastBotIndex].message = collected;
                        }
                        return updated;
                    });
                }
            };

            eventSource.onerror = (err) => {
                // console.error("SSE Error:", err);
                setIsAiThinking(false);
                eventSource.close();
            };


            // if (titleId) {
            //     const res = await request({
            //         url: `/ai/get-info?title_id=${titleId}`,
            //         method: "POST"
            //     });
            // }

            // if (botRes.success) {
            //     const botMessage: ChatMessage = {
            //         message: botRes?.answer ?? "No response received",
            //         sender: "bot",
            //         title_id: Number(currentTitleId)
            //     };
            //     setAllChats((prev) => [...prev, botMessage]);
            //     await sendResponse(botMessage.message, "bot", currentTitleId);
            //     await getAllChats(currentTitleId);
            // }
        } catch (err) {
            console.error("Error handling chat:", err);
            setIsAiThinking(false);

            setAllChats((prev) => [...prev, { message: "I am not able to find", sender: "bot", title_id: Number(titleId) }]);
            showAlert("Error", (err as Error).message || "Something went wrong", "error");
        }
        // finally {
        //     setIsAiThinking(false);
        // }
    }, [allChats, request, sendNewChat, sendResponse, titleId, userId, value, navigate]);

    /** Feedback handling */
    const sendFeedback = useCallback(async (chatId: string, feedback: 0 | 1 | null) => {
        try {
            await request({
                url: `/chatbot/likeChat`,
                method: "POST",
                body: { chat_id: chatId, user_id: userId, feedback },
            });
        } catch (err) {
            console.error("Error submitting feedback:", err);
        }
    }, [request, userId]);

    const handleLike = (chatId: string) => {
        const newStatus = feedbackMap[chatId] === 0 ? null : 0;
        setFeedbackMap((prev) => ({ ...prev, [chatId]: newStatus }));
        sendFeedback(chatId, newStatus);
    };

    const handleDislike = (chatId: string) => {
        const newStatus = feedbackMap[chatId] === 1 ? null : 1;
        setFeedbackMap((prev) => ({ ...prev, [chatId]: newStatus }));
        sendFeedback(chatId, newStatus);
    };

    /** Report a chat/query */
    const handleReport = async (data: string) => {
        try {
            const res = await request({
                url: `/support/addQuery`,
                method: "POST",
                body: { query: data, user_id: userId },
            });
            showAlert("Success", res?.success ?? "Reported successfully", "success");
            setReportData("");
        } catch (err) {
            console.error("Error reporting query:", err);
            showAlert("Error", (err as Error).message || "Failed to report", "error");
        }
    };

    useEffect(() => {
        if (userId) getSidebarData(userId);
    }, [userId, getSidebarData]);

    useEffect(() => {
        if (titleId) getAllChats(titleId);
    }, [userId, titleId, getAllChats]);

    const handleKeyDown = (e: React.KeyboardEvent<HTMLTextAreaElement>) => {
        if (e.key === 'Enter' && !e.shiftKey) {
            e.preventDefault()
            handleSubmit(e);
        }
    }

    const handleDeleteTitle = async (e: React.MouseEvent<HTMLButtonElement | HTMLDivElement>, id: number) => {
        e.stopPropagation();
        try {
            const res = await request({
                url: `/chatbot/deleteChatTitle`,
                method: "DELETE",
                body: { title_id: id },
            });

            if (res.error) {
                return showAlert("Error", res.error || "Failed to delete title", "error");
            }
            if (titleId === String(id)) {
                navigate(`/home/${userId}`);
            }
            getSidebarData(userId);
        } catch (err) {
            console.error("Error deleting title:", err);
            showAlert("Error", (err as Error).message || "Failed to delete title", "error");
        }
    }

    const handleChatReaction = async (chatId?: number, reaction?: number) => {
        try {
            const res = await request({
                url: `/chatbot/reaction`,
                method: "POST",
                body: { chat_id: chatId, user_id: userId, reaction },
            });

            if (res.error) {
                return showAlert("Error", res.error || "Failed to delete title", "error");
            }

            getAllChats(titleId);
        } catch (err) {
            console.error("Error deleting title:", err);
            showAlert("Error", (err as Error).message || "Failed to delete title", "error");
        }
    }

    const handleCopy = async (message: string, chatId: number) => {
        try {
            await navigator.clipboard.writeText(message);
            setCopiedChatId(chatId);
            setTimeout(() => setCopiedChatId(null), 1500);
        } catch (err) {
            console.error("Clipboard copy failed, using fallback:", err);

            const textarea = document.createElement("textarea");
            textarea.value = message;
            textarea.style.position = "fixed";
            textarea.style.left = "-9999px";
            textarea.style.top = "0";
            document.body.appendChild(textarea);
            textarea.focus();
            textarea.select();

            try {
                document.execCommand("copy");
                setCopiedChatId(chatId);
                setTimeout(() => setCopiedChatId(null), 1500);
            } catch (fallbackErr) {
                console.error("Fallback copy failed:", fallbackErr);
            }

            document.body.removeChild(textarea);
        }
    };


    const chatContainerRef = useRef<HTMLDivElement | null>(null);
    useEffect(() => {
        if (chatContainerRef.current) {
            chatContainerRef.current.scrollTop = chatContainerRef.current.scrollHeight;
        }
    }, [allChats]);

    return (
        <Flex h={'100vh'} bg={'#F2F2F2'} p={4} gap={2}>
            <Box w={'400px'} h={'100%'}>
                <Sidebar allTitles={allTitles}
                    onSelect={(id) => navigate(`/home/${userId}/${id}`)}
                    onNewChat={() => navigate(`/home/${userId}`)}
                    handleDeleteTitle={(e, id) => handleDeleteTitle(e, id)}
                />
            </Box>
            <Box w={'100%'} bg={'#fff'} h={"100%"} borderRadius={'20px'} p={4}>
                <Flex justifyContent={'space-between'}>
                    <Image src={logo} h={'40px'} w={'40px'} />
                    <Avatar.Root size={'sm'}>
                        <Avatar.Fallback name="Animesh Pradhan" />
                        <Avatar.Image src="https://bit.ly/sage-adebayo" />
                    </Avatar.Root>
                </Flex>

                <Flex mt={2} borderRadius={'10px'} h={"calc(100vh - 108px)"} bg={'#dee9f1'}>
                    {!titleId ? <Flex h={'100%'} alignItems={'center'} justifyContent={'center'} flexDir={'column'} w={'100%'}>
                        <Heading fontSize={'24px'} fontWeight={700} w={'50%'} textAlign={'center'}>Unleash AI with SmartProperty assistant and insights at your Fingertips</Heading>
                        <Text mt={4} w={'70%'} color={'gray.600'} textAlign={'center'}>Your intelligent real estate partner that understands your needs, recommends the best properties, and helps you book site visits instantly.</Text>

                        <Flex align="center" gap={2} mt={4} w={'70%'}>
                            <Textarea
                                value={value}
                                onChange={(e) => setValue(e.target.value)}
                                onKeyDown={handleKeyDown}
                                placeholder="Ask me anything..."
                                rows={3}
                                w="100%"
                                bg="#fff"
                                borderRadius="10px"
                                border="1px solid #005392"
                                resize="none"
                                _focus={{
                                    outline: 'none',
                                    borderColor: '#005392',
                                    boxShadow: '0 0 8px 2px rgba(0, 83, 146, 0.7)',
                                }}
                                _hover={{
                                    boxShadow: '0 0 6px 1px rgba(0, 83, 146, 0.5)',
                                }}
                            />

                            <IconButton onClick={handleSubmit} _hover={{ bg: '#004276' }} _active={{ bg: '#00365f' }} bg="#005392" color="#fff" borderRadius="full" aria-label="Search database">
                                <IoSend />
                            </IconButton>
                        </Flex>
                    </Flex> : <Flex p={4} flexDir={'column'} w={'100%'}>
                        <Flex flex={1} gap={2} h={'100%'} overflow={'auto'} flexDir={'column'} className="scroll-container" ref={chatContainerRef}>
                            {allChats.map((chat, index) => {
                                if (chat.sender === "bot" && chat.message.trim() === "") return null;
                                return (
                                    <Flex key={index} flexDir={'column'}>
                                        <Box alignSelf={chat.sender === "user" ? "flex-end" : "flex-start"} bg={chat.sender === "user" ? "#fff" : "#005392"} color={chat.sender === "user" ? "#000" : "white"} borderRadius="20px" p="10px" maxW="60%" my="8px" boxShadow="md">
                                            {chat.sender === "bot" && index === allChats.length - 1 ? (
                                                <Box>
                                                    {chat?.message
                                                        ?.split(/\n+/)
                                                        .filter(line => line.trim() !== '')
                                                        .map((line, idx) => {
                                                            const withIndent = line.replace(/\t/g, '\u00A0\u00A0\u00A0\u00A0');

                                                            // simplified and reliable URL regex
                                                            const urlRegex = /(https?:\/\/\S+)/;

                                                            // ALWAYS capture URL parts correctly
                                                            const parts = withIndent.split(/(\*\*.*?\*\*|\*.*?\*|https?:\/\/\S+)/);

                                                            const parsedLine = parts.map((part, i) => {
                                                                if (!part) return null;

                                                                const trimmed = part.trim();

                                                                // detect URL
                                                                if (urlRegex.test(trimmed)) {
                                                                    let clean = trimmed.replace(/[")<>]+$/g, '');
                                                                    return (
                                                                        <a key={i} href={clean} target="_blank"
                                                                            rel="noopener noreferrer"
                                                                            style={{ color: '#fff', textDecoration: 'underline', fontWeight: 'bold' }}
                                                                        >
                                                                            Click Here
                                                                        </a>
                                                                    );
                                                                }

                                                                // bold
                                                                const boldMatch = part.match(/^\*\*(.*?)\*\*$/);
                                                                if (boldMatch) {
                                                                    return (
                                                                        <Text as="span" fontWeight="bold" key={i}>
                                                                            {boldMatch[1]}
                                                                        </Text>
                                                                    );
                                                                }


                                                                // italic
                                                                if (trimmed.startsWith('*') && trimmed.endsWith('*')) {
                                                                    return (
                                                                        <Text as="span" fontStyle="italic" key={i}>
                                                                            {trimmed.slice(1, -1)}
                                                                        </Text>
                                                                    );
                                                                }



                                                                return <Text as="span" key={i}>{part}</Text>;
                                                            });

                                                            return (
                                                                <Flex key={idx} flexDir={'column'}>
                                                                    <Text mt={idx !== 0 ? 3 : 0}>{parsedLine}</Text>
                                                                </Flex>
                                                            );
                                                        })}
                                                </Box>

                                            ) : (
                                                <Box>
                                                    <Box>
                                                        {chat?.message
                                                            ?.split(/\n+/)
                                                            .filter(line => line.trim() !== '')
                                                            .map((line, idx) => {
                                                                const withIndent = line.replace(/\t/g, '\u00A0\u00A0\u00A0\u00A0');

                                                                // simplified and reliable URL regex
                                                                const urlRegex = /(https?:\/\/\S+)/;

                                                                // ALWAYS capture URL parts correctly
                                                                const parts = withIndent.split(/(https?:\/\/\S+)/);
                                                                const parsedLine = parts.map((part, i) => {
                                                                    if (!part) return null;

                                                                    const trimmed = part.trim();

                                                                    // detect URL
                                                                    if (urlRegex.test(trimmed)) {
                                                                        let clean = trimmed.replace(/[")<>]+$/g, '');

                                                                        // console.log('here', clean);

                                                                        return (
                                                                            <a key={i} href={clean} target="_blank"
                                                                                rel="noopener noreferrer"
                                                                                style={{ color: '#fff', textDecoration: 'underline', fontWeight: 'bold' }}
                                                                            >
                                                                                Click Here
                                                                            </a>
                                                                        );
                                                                    }

                                                                    // bold
                                                                    if (trimmed.startsWith('**') && trimmed.endsWith('**')) {
                                                                        return (
                                                                            <Text as="span" fontWeight="bold" key={i}>
                                                                                {trimmed.slice(2, -2)}
                                                                            </Text>
                                                                        );
                                                                    }

                                                                    // italic
                                                                    if (trimmed.startsWith('*') && trimmed.endsWith('*')) {
                                                                        return (
                                                                            <Text as="span" fontStyle="italic" key={i}>
                                                                                {trimmed.slice(1, -1)}
                                                                            </Text>
                                                                        );
                                                                    }



                                                                    return <Text as="span" key={i}>{part}</Text>;
                                                                });

                                                                return (
                                                                    <Box key={idx} mt={idx !== 0 ? 3 : 0}>
                                                                        {parsedLine}
                                                                    </Box>
                                                                );
                                                            })}
                                                    </Box>
                                                </Box>
                                            )}
                                        </Box>
                                        {chat.sender === "bot" && <Flex gap={2} alignItems={'center'}>
                                            <IconButton variant={'ghost'}
                                                onClick={() => chat.id && handleCopy(chat.message, chat.id)}
                                                size={'2xs'}
                                            >
                                                {copiedChatId === chat.id ? <IoIosCopy /> : <MdContentCopy />}
                                            </IconButton>

                                            {chat.reaction === null ? <>
                                                <IconButton variant={'ghost'} size={'2xs'} disabled={loading} onClick={() => handleChatReaction(chat.id, 0)}>
                                                    <MdThumbDownOffAlt />
                                                </IconButton>
                                                <IconButton variant={'ghost'} size={'2xs'} disabled={loading} onClick={() => handleChatReaction(chat.id, 1)}>
                                                    <MdThumbUpOffAlt />
                                                </IconButton>
                                            </> : <IconButton variant={'ghost'} size={'2xs'}>
                                                {chat.reaction === 0 ? <MdThumbDown /> : <MdThumbUp />}
                                            </IconButton>}

                                            <IconButton variant={'ghost'} size={'2xs'}>
                                                <MdFeedback />
                                            </IconButton>
                                        </Flex>}
                                    </Flex>
                                )
                            })}

                            {isAiThinking && (
                                <Flex flexDir="column" alignSelf="flex-start" my="8px">
                                    <Box color="white" boxShadow="md" ml={'30px'}>
                                        <Box className="loader" />
                                    </Box>
                                </Flex>
                            )}
                        </Flex>

                        <Flex align="center" gap={2} mt={4} w={'100%'}>
                            <Textarea
                                value={value}
                                onChange={(e) => setValue(e.target.value)}
                                onKeyDown={handleKeyDown}
                                placeholder="Ask me anything..."
                                rows={3}
                                w="100%"
                                bg="#fff"
                                borderRadius="10px"
                                border="1px solid #005392"
                                resize="none"
                                _focus={{
                                    outline: 'none',
                                    borderColor: '#005392',
                                    boxShadow: '0 0 8px 2px rgba(0, 83, 146, 0.7)',
                                }}
                                _hover={{
                                    boxShadow: '0 0 6px 1px rgba(0, 83, 146, 0.5)',
                                }}
                            />

                            <IconButton disabled={isAiThinking} onClick={handleSubmit} _hover={{ bg: '#004276' }} _active={{ bg: '#00365f' }} bg="#005392" color="#fff" borderRadius="full" aria-label="Search database">
                                <IoSend />
                            </IconButton>
                        </Flex>
                    </Flex>}
                </Flex>
            </Box >
        </Flex >
    );
};

export default Home;




const Sidebar: React.FC<SidebarProps> = ({ allTitles, onSelect, onNewChat, handleDeleteTitle }) => {
    const navigate = useNavigate();
    return (
        <Flex h={'100%'} w="100%" bg="#d9e9f4ff" borderRadius={'20px'} p={4} flexDirection="column" gap={4}>

            <Flex alignItems={'center'} justifyContent={'space-between'}>
                <Flex alignItems={'center'} gap={2}>
                    <IconButton borderRadius={'full'} size={'xs'} bg={'#0000001A'} color={'#000'}><GoSidebarExpand /></IconButton>
                    <Text fontSize={'20px'} fontWeight={600}>Chats</Text>
                </Flex>

                <IoSearchSharp />
            </Flex>


            <Button onClick={onNewChat} colorScheme="blue" w="100%" borderRadius="full" bg={'blue.700'} _hover={{ bg: 'blue.800' }}>
                <FiPlus /> New Chat
            </Button>

            <Separator borderColor={'blue.700'} alignSelf={'center'} w={'150px'} />
            {/* Recent Chats */}

            <Flex alignItems={'center'} gap={4}>
                <Box fontSize={'24px'} color="gray.600">
                    <LuMessageSquareText />
                </Box>
                <Text fontWeight="semibold" mb={2} color="gray.600">Recent Chats</Text>
            </Flex>

            <Flex flexDir={'column'} gap={1} h={'80%'} overflowY={'auto'} className="scroll-container">
                {allTitles.map((chat) => (
                    <Flex justifyContent={'space-between'} alignItems={'center'} key={chat.id} borderRadius="md" _hover={{ bg: "gray.100" }} cursor="pointer" p={1} onClick={() => onSelect(chat.id)}>
                        <Box>
                            <Text fontSize="13px" color="gray.800" fontWeight="medium">
                                {chat.title}
                            </Text>
                            <Text fontSize="12px" color="gray.500">
                                {new Date(chat.created_at).toLocaleString(undefined, {
                                    day: "numeric",
                                    month: "short",
                                    year: "numeric",
                                    hour: "2-digit",
                                    minute: "2-digit",
                                })}
                            </Text>
                        </Box>

                        <IconButton size={'2xs'} variant={'ghost'} onClick={(e) => handleDeleteTitle(e, Number(chat.id))} color="red.500">
                            <MdDelete />
                        </IconButton>

                    </Flex>
                ))}
            </Flex>

            <Separator />

            <VStack align="stretch" gap={2} h={'50px'} mb={4}>
                <Button size={'xs'} variant="ghost" justifyContent="flex-start"> <FiSettings /> Settings</Button>
                <Button size={'xs'} variant="ghost" justifyContent="flex-start"><FiHelpCircle /> Help & Support</Button>
            </VStack>
        </Flex>
    )
}
