import React from 'react';
import { Box, VStack, Text, Button, Input, Icon, Avatar, Separator, Flex, IconButton, Image, Heading, Textarea } from "@chakra-ui/react"
import { FiPlus, FiSettings, FiHelpCircle, FiSearch } from "react-icons/fi"
import { GoSidebarExpand } from 'react-icons/go';
import { IoSearchSharp } from 'react-icons/io5';
import { PiChatsBold } from 'react-icons/pi';
import { LuMessageSquareText } from 'react-icons/lu';

// @ts-ignore
import logo from "../assets/logo.png";
import { useParams } from 'react-router';

const Home: React.FC = () => {
    const { titleId, userId } = useParams();



    return (
        <Flex h={'100vh'} bg={'#F2F2F2'} p={4} gap={2}>
            <Box w={'400px'} h={'100%'}>
                <Sidebar />
            </Box>
            <Box w={'100%'} bg={'#fff'} h={"100%"} borderRadius={'20px'} p={4}>
                <Flex justifyContent={'space-between'}>
                    <Image src={logo} h={'40px'} w={'40px'} />
                    <Avatar.Root size={'sm'}>
                        <Avatar.Fallback name="Animesh Pradhan" />
                        <Avatar.Image src="https://bit.ly/sage-adebayo" />
                    </Avatar.Root>
                </Flex>

                <Flex mt={2} borderRadius={'10px'} h={"calc(100ch - 238px)"} bg={'#dee9f1'}>
                    {!titleId && <Flex h={'100%'} alignItems={'center'} justifyContent={'center'} flexDir={'column'} w={'100%'}>
                        <Heading fontSize={'24px'} fontWeight={700} w={'50%'} textAlign={'center'}>Unleash AI with CiplaGPT Smarter Ideas and insights at your Fingertips</Heading>
                        <Text mt={4} w={'70%'} color={'gray.600'} textAlign={'center'}>Turn imagination into impact with ChatGPT’s AI built to unlock endless possibilities and shape your ideas into intelligent results.</Text>

                        <Textarea w={'70%'} bg={'#fff'} borderRadius={'10px'} border="1px solid #005392" mt={4} rows={5} placeholder='Ask me anything...' _focus={{
                            outline: 'none',
                            borderColor: '#005392', // optional border color
                            boxShadow: '0 0 8px 2px rgba(0, 83, 146, 0.7)', // glowing effect
                        }}
                            _hover={{
                                boxShadow: '0 0 6px 1px rgba(0, 83, 146, 0.5)',
                            }}
                        />
                    </Flex>}
                </Flex>
            </Box>
        </Flex>
    );
};

export default Home;



function Sidebar() {
    return (
        <Flex h={'100%'} w="100%" bg="#d9e9f4ff" borderRadius={'20px'} p={4} flexDirection="column" gap={4}>

            <Flex alignItems={'center'} justifyContent={'space-between'}>
                <Flex alignItems={'center'} gap={2}>
                    <IconButton borderRadius={'full'} size={'xs'} bg={'#0000001A'} color={'#000'}><GoSidebarExpand /></IconButton>
                    <Text fontSize={'20px'} fontWeight={600}>Chats</Text>
                </Flex>

                <IoSearchSharp />
            </Flex>


            <Button colorScheme="blue" w="100%" borderRadius="full" bg={'blue.700'} _hover={{ bg: 'blue.800' }}>
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

            <Flex flexDir={'column'} gap={2} h={'80%'} overflowY={'auto'} className="scroll-container">
                {Array.from({ length: 6 }).map((_, i) => (
                    <Box key={i} borderRadius="md" _hover={{ bg: "gray.200" }} cursor="pointer">
                        <Text fontSize="sm" color="gray.700">Give me unique name logo for my company with a better design and content</Text>
                        <Text fontSize="xs" color="gray.500">Today, 4:22 am</Text>
                    </Box>
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
