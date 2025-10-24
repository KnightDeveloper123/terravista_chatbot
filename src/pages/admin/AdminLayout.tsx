import { Box, Flex } from '@chakra-ui/react';
import React from 'react';
import { Outlet } from 'react-router';
import Navbar from './Navbar';
import Sidebar from './Sidebar';

const AdminLayout: React.FC = () => {
    return (
        <Box h={'100vh'}>
            <Box h={'60px'} boxShadow="0 2px 4px rgba(0, 0, 0, 0.1)">
                <Navbar />
            </Box>

            <Flex h={'calc(100vh - 75px)'} mt={'5px'}>
                <Box w={'240px'}>
                    <Sidebar />
                </Box>

                <Box p={4} h={'100%'} bg={'#fafafa'} w={'100%'}>
                    <Outlet />
                </Box>
            </Flex>
        </Box>
    );
};

export default AdminLayout;