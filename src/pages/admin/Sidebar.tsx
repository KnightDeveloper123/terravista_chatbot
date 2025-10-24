import { Box, Flex, Icon, Text, VStack } from '@chakra-ui/react';
import React from 'react';
import { HiOutlineUsers } from 'react-icons/hi';
import { MdOutlineDashboard } from 'react-icons/md';
import { Link } from 'react-router';

const Sidebar: React.FC = () => {

    const adminNavbar = [
        { title: "Dashboard", url: "/admin/dashboard", icon: MdOutlineDashboard, },
        // { title: "Courses", url: "/admin/courses", icon: SiWikibooks, },
        { title: "Users", url: "/admin/users", icon: HiOutlineUsers, },
        // { title: "Purchase History", url: "/admin/purchases", icon: RiCustomerService2Line, },
        // { title: "Exams", url: "/admin/exams", icon: PiExam, },
        // { title: "Queries", url: "/queries", icon: RiCustomerService2Line, },
        // { title: "Customer Support", url: "/customer_request", icon: PiUsersThree, },
        // { title: "Notifications", url: "/notifications", icon: AiOutlineBell, },
        // { title: "Reports", url: "/reports", icon: HiOutlineDocumentReport, },
        // { title: "Settings", url: "/settings", icon: IoSettingsOutline, },
    ];


    return (
        <Box width={'100%'} h='100%' bg="#fff" p={4}>
            <VStack gap={2} align="stretch">
                {adminNavbar.map((item, index) => {
                    const isActive = location.pathname === item.url;
                    return (
                        <Link key={index} to={item.url}>
                            <Box
                                borderRadius="md"
                                display="flex"
                                alignItems="center"
                                bg={isActive ? "#004AAD" : "#fff"}
                                fontWeight={isActive ? "bold" : "semibold"}
                                color={isActive ? "#fff" : "#606060"}
                                transition="all 0.25s ease"
                                px={3}
                                py={2}
                                boxShadow={isActive ? "0 4px 10px rgba(0, 74, 173, 0.2)" : "0 1px 2px rgba(0,0,0,0.05)"}
                                _hover={{
                                    transform: "translateY(-2px)",
                                    boxShadow: "0 4px 10px rgba(0, 0, 0, 0.1)",
                                    bg: isActive ? "#003580" : "#f8f9fa",
                                }}
                                cursor="pointer"
                            >
                                <Flex align="center" gap={3} w="100%">
                                    <Icon
                                        as={item.icon}
                                        fontSize={22}
                                        color={isActive ? "#fff" : "#004AAD"}
                                        transition="color 0.2s"
                                    />
                                    <Text fontSize="14px" letterSpacing="0.5px" textTransform={'capitalize'}>
                                        {item.title}
                                    </Text>
                                </Flex>
                            </Box>
                        </Link>
                    );
                })}
            </VStack>
        </Box>
    );
};

export default Sidebar;